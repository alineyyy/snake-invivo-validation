"""Small configuration layer for the real-data activation-map notebook."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fmri_glm_toolkit import GlobalConfig, ReconSpec


RESULTS_ROOT = Path("/volatile/Caini/mnt/topaze/scratch/results_3T_invivo")
MATLAB_CMD = (
    "/volatile/Caini/stimulate/spm/spm_standalone/run_spm25.sh "
    "/usr/local/MATLAB_Runtime/R2024b/ script"
)
DEFAULT_T1 = "/volatile/Caini/000005_t1-mpr-moco-tra-iso1mm-moco_t1_mpr-moco_tra_iso1mm_20260224101132_5.nii"

# Acquisition affine indexed by visit. Values were collected from
# GLM_master_realdata.ipynb (old cell 13).
AFFINES = {
    "1219": np.array([
        [-1.46897663e-11, 3.0, 0.0, -96.0],
        [3.0, 1.46897663e-11, 0.0, -73.8785954],
        [0.0, 0.0, -3.0, 97.7907333],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "0210": np.array([
        [-1.46897663e-11, 3.0, 0.0, -96.0],
        [3.0, 1.46897663e-11, 0.0, -80.1404142],
        [0.0, 0.0, -3.0, 102.639393],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "0224": np.array([
        [-1.46897663e-11, 3.0, 0.0, -96.0],
        [3.0, 1.46897663e-11, 0.0, -77.2672806],
        [0.0, 0.0, -3.0, 79.5817680],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "0306": np.array([
        [-1.46897663e-11, 3.0, 0.0, -96.0],
        [3.0, 1.46897663e-11, 0.0, -75.2505684],
        [0.0, 0.0, -3.0, 94.5823860],
        [0.0, 0.0, 0.0, 1.0],
    ]),
}


def run_number(label: str) -> int:
    """Return x from a label starting with Rx."""
    match = re.match(r"^R(\d+)(?:_|$)", label)
    if not match:
        raise ValueError(f"Label must start with Rx, for example R4_36_cs: {label!r}")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError(f"R must be positive: {label!r}")
    return value


def make_config(
    visit: str = "0224",
    *,
    cut_coords=(20, -56, 6),
    alpha=1e-3,
    cluster_th=30,
    n_jobs=4,
) -> tuple[GlobalConfig, np.ndarray]:
    """Build global output/QC settings and select the visit affine."""
    if visit not in AFFINES:
        raise ValueError(f"Unknown visit {visit!r}; choose one of {sorted(AFFINES)}")
    cfg = GlobalConfig(
        father_out=str(RESULTS_ROOT / f"nifti_{visit}"),
        default_t1_path=DEFAULT_T1,
        matlab_cmd=MATLAB_CMD,
        tr=1.0,  # Per-run TR is stored on ReconSpec; this is only a fallback.
        alpha_fpr_default=float(alpha),
        cluster_th_default=int(cluster_th),
        cut_coords=tuple(cut_coords),
        n_jobs=int(n_jobs),
        spm_jobtype="estimate",
        save_design_matrix=True,
    )
    return cfg, AFFINES[visit].copy()


def make_specs(
    runs: Mapping[str, str | Sequence[str] | Path],
    *,
    visit: str = "0224",
    overwrite=False,
    scale=1.0,
) -> list[ReconSpec]:
    """Create specs from ``label -> date glob(s)`` or ``label -> direct path``.

    For a date selection the input is resolved below recon_VISIT/label and the
    suffix is always identical to the output label. A direct Path selects one
    directory or .npy file explicitly.
    """
    recon_base = RESULTS_ROOT / f"recon_{visit}"
    specs = []
    for label, selection in runs.items():
        r = run_number(label)
        if isinstance(selection, Path):
            path = selection
        else:
            dates = [selection] if isinstance(selection, str) else list(selection)
            if not dates:
                raise ValueError(f"No date/data selection supplied for {label}")
            path = {"base": str(recon_base), "dates": dates, "suffix": label}
        specs.append(ReconSpec(
            path=path,
            label=label,
            overwrite=bool(overwrite),
            scale=float(scale),
            n_scans=120 * r,
            tr=2.0 / r,
            bg_mode="mean_func",
            use_bg_complex=False,
            bg_policy="none",
        ))
    return specs


def preview_specs(specs: Sequence[ReconSpec]) -> None:
    """Print resolved directories and fail before GLM if a selection is empty."""
    from fmri_glm_toolkit import _expand_recon_path_spec

    print(f"{'label':32} {'TR':>6} {'scans':>7}  selected data")
    print("-" * 100)
    for spec in specs:
        paths = _expand_recon_path_spec(spec.path)
        print(f"{spec.label:32} {spec.tr:6g} {spec.n_scans:7d}  {len(paths)} path(s)")
        for path in paths:
            print(f"{'':50}  {path}")
