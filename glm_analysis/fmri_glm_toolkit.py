# fmri_glm_batch.py
import os
import re
import json
import time
import shutil
import traceback
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import nibabel as nib

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from nipype.interfaces import spm
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.image import resample_to_img
from nilearn.plotting import plot_design_matrix, plot_stat_map



import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

import numpy as np
import pandas as pd

def make_block_events_variable(n_scans, tr, block_durations, rest_duration=30.0):
    """
    Generate block-design events for fMRI, allowing variable block durations.
    - block_durations: list or array of durations for each block (in seconds)
    """
    total_time = n_scans * tr
    onsets = []
    current_onset = 0.0

    # Generate onsets block by block
    for i, dur in enumerate(block_durations):
        if current_onset + dur <= total_time:
            onsets.append(current_onset)
            current_onset += dur + rest_duration
        else:
            break  # stop if next block exceeds total scan duration

    events = pd.DataFrame({
        'onset': onsets,
        'duration': block_durations[:len(onsets)],
        'trial_type': ['checkerboard'] * len(onsets)
    })

    # Diagnostics
    print(f"[make_block_events_variable] total_time={total_time:.1f}s, n_blocks={len(onsets)}")
    if len(onsets) > 0:
        print(f" first onset={onsets[0]:.3f}s, last onset={onsets[-1]:.3f}s, "
              f"last block end={onsets[-1] + block_durations[len(onsets)-1]:.3f}s")

    return events


def run_first_level_single_session(fmri_img, motion_params, n_scans, tr,
                                   block_duration=30, rest_duration=30.0):
    """
    Full single-session first-level GLM pipeline for block design.
    Returns (z_map, design_matrix, glm, events).
    """
    # Frame times
    frame_times = np.arange(n_scans) * tr

    # Generate events
    block_durations = [30, 30, 30, 30]
    rest_duration = 30.0

    events = make_block_events_variable(n_scans, tr, block_durations, rest_duration)

    # Motion parameters check
    add_regs = np.asarray(motion_params)
    if add_regs.ndim != 2:
        raise ValueError("motion_params must have shape (n_scans, n_params)")
    if add_regs.shape[0] != n_scans:
        raise ValueError(f"motion_params rows ({add_regs.shape[0]}) must match n_scans ({n_scans})")

    add_reg_names = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']

    # Design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model='spm',
        drift_model='polynomial',
        drift_order=5,
        add_regs=add_regs,
        add_reg_names=add_reg_names
    )

    # Fit GLM
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model='spm',
        drift_model='polynomial',
        drift_order=5
    )
    glm = glm.fit(fmri_img, design_matrices=design_matrix)

    # Contrast: checkerboard > baseline
    if 'checkerboard' not in design_matrix.columns:
        raise ValueError("No 'checkerboard' column found in design matrix. Check events.")
    contrast_vec = (design_matrix.columns == 'checkerboard').astype(float)
    print(contrast_vec)
    res = glm.compute_contrast(contrast_vec, output_type='all')
    z_map = res['z_score']

    return z_map, design_matrix, glm, events

# -------------------------
# Dataclasses: config + spec
# -------------------------
@dataclass
class GlobalConfig:
    father_out: str
    default_t1_path: str
    matlab_cmd: str
    tr: float = 1.0
    alpha_fpr_default: float = 1e-3
    cluster_th_default: int = 60
    cut_coords: Tuple[float, float, float] = (20, -56, 6)
    n_jobs: int = 4

    # SPM behavior: "estimate" (only rp_*.txt) is safest/closest to your earlier usage
    spm_jobtype: str = "estimate"

    # For nilearn resample warnings control
    force_resample: bool = True
    copy_header: bool = True

    # Save design matrix per experiment folder
    save_design_matrix: bool = True


@dataclass
class ReconSpec:
    path: Any
    label: Optional[str] = None
    # when True, remove the existing output folder for this label and rerun
    overwrite: bool = False
    
    # scale applied to npy before further processing
    # path can be a single .npy, a directory of xxx_0.npy, xxx_1.npy, ...
    # a list of such paths, or a dict:
    #   {"base": ".../recon_1219", "date": "2026-04-27_15-39",
    #    "suffixes": ["R1_288_cg_orc", "R1_298_cg"]}
    #   {"base": ".../recon_1219", "dates": ["2026-04-27_15*", "2026-04-27_17*"],
    #    "suffix": "R1_288_cg_orc"}
    #   {"base": ".../recon_1219", "date": "2026-04-27_15-*",
    #    "suffix": "R1_288_cg_orc"}
    # (set to 40000 if your INR outputs need it; set to 1.0 if already scaled)
    scale: float = 1.0
    tr: float = 1.0
    # if known, pass 240; if None, auto-infer time axis
    n_scans: Optional[int] = None
    # if set, force the recon time axis before converting to XYZT
    recon_time_axis: Optional[int] = None

    # background underlay used for activation overlay/resample target
    # "t1": use bg_t1_path if provided, else cfg.default_t1_path
    # "mean_func": use last time frame of recon.nii as background (in recon space)
    bg_mode: str = "t1"
    bg_t1_path: Optional[str] = None

    # recon input format handling:
    # supports complex64, real/imag channel (..,2), magnitude float
    # optional complex background:
    use_bg_complex: bool = False
    bg_complex_npy: Optional[str] = None

    # how to combine background if use_bg_complex=True:
    # "none" | "complex_add_then_abs" | "abs_add"
    bg_policy: str = "none"


@dataclass
class OneReconResult:
    label: str
    recon_path: str
    out_dir: str
    status: str
    T: Optional[int] = None
    threshold: Optional[float] = None
    n_supra: Optional[int] = None
    max_z: Optional[float] = None
    error: Optional[str] = None


# -------------------------
# Small helpers
# -------------------------
def _configure_spm(matlab_cmd: str):
    os.environ["SPMMCRCMD"] = matlab_cmd
    os.environ["FORCE_SPMMCR"] = "1"
    spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
    _ = spm.SPMCommand().version


def _logger(out_dir: Path):
    log_path = out_dir / "run.log"

    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")
    return log


def _maybe_reset_out_dir(out_dir: Path, father_out: Path, overwrite: bool) -> bool:
    out_dir = Path(out_dir)
    father_out = Path(father_out)
    if not overwrite or not out_dir.exists():
        return False
    if not out_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {out_dir}")

    out_dir_resolved = out_dir.resolve(strict=False)
    father_resolved = father_out.resolve(strict=False)
    try:
        out_dir_resolved.relative_to(father_resolved)
    except ValueError as exc:
        raise ValueError(f"Refusing to overwrite path outside father_out: {out_dir}") from exc
    if out_dir_resolved == father_resolved:
        raise ValueError(f"Refusing to overwrite father_out root directly: {father_out}")

    shutil.rmtree(out_dir)
    return True


def infer_label_from_path(p: Path) -> str:
    if p.is_dir():
        return p.name.replace("_", "")

    stem = p.stem
    # reconstruction_2960 / reconstructed_0
    m = re.search(r"(reconstruction|reconstructed)_(\d+)$", stem)
    it = m.group(2) if m else None

    if p.parent.name == "images" and p.parent.parent is not None:
        method_dir = p.parent.parent.name
        parts = method_dir.split("_")
        method_short = "".join(parts[:2]) if len(parts) >= 2 else method_dir.replace("_", "")
        return f"{method_short}iter{it}" if it is not None else method_short

    base = p.parent.name.replace("_", "")
    if it is not None:
        return f"{base}iter{it}"
    return stem


def _path_spec_for_log(path_spec: Any) -> str:
    if isinstance(path_spec, dict):
        return json.dumps(path_spec, ensure_ascii=False)
    if isinstance(path_spec, (list, tuple)):
        return "[" + ", ".join(str(p) for p in path_spec) + "]"
    return str(path_spec)


def infer_label_from_recon_path_spec(path_spec: Any) -> str:
    if isinstance(path_spec, dict):
        suffixes = path_spec.get("suffixes", path_spec.get("runs", path_spec.get("run_names")))
        if suffixes is None:
            suffixes = path_spec.get("suffix", path_spec.get("run", path_spec.get("run_name")))
        if suffixes is not None:
            if isinstance(suffixes, str):
                return suffixes.replace("_", "")
            return "_".join(str(s).replace("_", "") for s in suffixes)
        if "paths" in path_spec:
            return infer_label_from_recon_path_spec(path_spec["paths"])
    if isinstance(path_spec, (list, tuple)):
        if not path_spec:
            return "empty_recon_spec"
        return infer_label_from_path(Path(path_spec[0]))
    return infer_label_from_path(Path(path_spec))


def _infer_time_axis(shape: Tuple[int, ...], n_scans_hint: Optional[int]) -> Tuple[int, int]:
    axes = list(range(len(shape)))
    if n_scans_hint is not None:
        matches = [ax for ax in axes if shape[ax] == n_scans_hint]
        if len(matches) == 1:
            ax = matches[0]
            return ax, shape[ax]
        if len(matches) > 1:
            # ambiguous, prefer last
            ax = matches[-1]
            return ax, shape[ax]

    # heuristic
    candidates = [ax for ax in axes if 10 <= shape[ax] <= 2000]
    if not candidates:
        return 0, shape[0]
    if (len(shape) - 1) in candidates:
        ax = len(shape) - 1
        return ax, shape[ax]
    return candidates[0], shape[candidates[0]]


def _moveaxis_to_last(a: np.ndarray, axis: int) -> np.ndarray:
    if axis == a.ndim - 1:
        return a
    return np.moveaxis(a, axis, -1)


def _moveaxis_to_penultimate(a: np.ndarray, axis: int) -> np.ndarray:
    target = a.ndim - 2
    if axis == target:
        return a
    return np.moveaxis(a, axis, target)


def _parse_indexed_npy_name(p: Path) -> Optional[Tuple[str, int]]:
    m = re.match(r"(.+?)_(\d+)$", p.stem)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def _has_glob_chars(s: str) -> bool:
    return any(ch in s for ch in "*?[")


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _expand_run_dirs(base: Path, suffix: str, date_patterns: List[Any]) -> List[Path]:
    suffix = str(suffix)
    run_parent = base / suffix
    run_dirs: List[Path] = []
    skipped_dirs: List[Path] = []
    for date_pattern in date_patterns:
        pattern = f"{date_pattern}_{suffix}"
        if _has_glob_chars(str(pattern)):
            matches = sorted(p for p in run_parent.glob(str(pattern)) if p.is_dir())
        else:
            candidate = run_parent / str(pattern)
            matches = [candidate] if candidate.is_dir() else []

        for match in matches:
            # The downstream reader only consumes top-level .npy files, so reject
            # empty/incomplete run directories here instead of failing later.
            if any(p.is_file() and p.suffix == ".npy" for p in match.iterdir()):
                run_dirs.append(match)
            else:
                skipped_dirs.append(match)

    # Overlapping patterns (for example a whole day plus one hour) must not load
    # the same run twice.  Dict/list order is retained for intentional merging.
    run_dirs = list(dict.fromkeys(run_dirs))
    if skipped_dirs:
        print("Skipping recon directories without .npy files:")
        for skipped in dict.fromkeys(skipped_dirs):
            print(f"  - {skipped}")
    return run_dirs


def _expand_recon_path_spec(npy_path: Any) -> List[Path]:
    if isinstance(npy_path, str) and npy_path.lstrip().startswith("{"):
        try:
            parsed = ast.literal_eval(npy_path)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return _expand_recon_path_spec(parsed)

    if isinstance(npy_path, (list, tuple)):
        paths = []
        for item in npy_path:
            paths.extend(_expand_recon_path_spec(item))
        return paths

    if isinstance(npy_path, dict):
        if "paths" in npy_path:
            return _expand_recon_path_spec(npy_path["paths"])

        base = npy_path.get("base", npy_path.get("root"))
        dates = npy_path.get("dates", npy_path.get("prefixes"))
        if dates is None:
            dates = npy_path.get("date", npy_path.get("prefix"))
        suffixes = npy_path.get("suffixes", npy_path.get("runs", npy_path.get("run_names")))
        if suffixes is None:
            suffixes = npy_path.get("suffix", npy_path.get("run", npy_path.get("run_name")))
        if base is None or dates is None or suffixes is None:
            raise ValueError(
                "Dict recon path must use either {'paths': [...]} or "
                "{'base': ..., 'date'/'dates': ..., 'suffix'/'suffixes': ...}"
            )
        date_patterns = _as_list(dates)
        run_dirs: List[Path] = []
        for suffix in _as_list(suffixes):
            run_dirs.extend(_expand_run_dirs(Path(base), str(suffix), date_patterns))
        run_dirs = list(dict.fromkeys(run_dirs))
        if not run_dirs:
            requested_dates = ", ".join(str(item) for item in date_patterns)
            requested_suffixes = ", ".join(str(item) for item in _as_list(suffixes))
            raise FileNotFoundError(
                f"No recon run directories containing .npy files found under {base}; "
                f"date pattern(s): {requested_dates}; suffix(es): {requested_suffixes}"
            )
        return run_dirs

    return [Path(npy_path)]


def _resolve_single_recon_npy_files(npy_path: str | Path) -> List[Path]:
    p = Path(npy_path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Recon path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Recon path must be a .npy file or directory: {p}")

    npy_files = sorted(q for q in p.iterdir() if q.is_file() and q.suffix == ".npy")
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in directory: {p}")

    indexed = []
    ignored = []
    for q in npy_files:
        parsed = _parse_indexed_npy_name(q)
        if parsed is None:
            ignored.append(q.name)
            continue
        prefix, idx = parsed
        indexed.append((prefix, idx, q))

    if indexed:
        prefixes = sorted({prefix for prefix, _, _ in indexed})
        if len(prefixes) > 1:
            raise ValueError(
                f"Found multiple indexed .npy prefixes in {p}: {prefixes}. "
                "Please keep only one sequence in this directory."
            )

        indices = [idx for _, idx, _ in indexed]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate indexed .npy suffix detected in {p}: {indices}")

        ordered = [q for _, _, q in sorted(indexed, key=lambda item: item[1])]
        if ignored:
            print(f"Ignoring non-indexed npy files in {p}: {ignored}")
        return ordered

    if len(npy_files) == 1:
        return npy_files

    raise ValueError(
        f"Directory {p} contains multiple .npy files but none match the xxx_0.npy pattern."
    )


def _resolve_recon_npy_files(npy_path: Any) -> List[Path]:
    roots = _expand_recon_path_spec(npy_path)
    files: List[Path] = []
    for root in roots:
        files.extend(_resolve_single_recon_npy_files(root))

    for q in files:
        parsed = _parse_indexed_npy_name(q)
        if parsed is None:
            if len(files) == 1:
                return files
            raise ValueError(
                f"Cannot combine multiple recon paths because {q} does not match xxx_0.npy pattern."
            )
    # Each directory is already numerically sorted by
    # _resolve_single_recon_npy_files.  Preserve directory order so multiple
    # hours concatenate as hour_1(_0, _1, ...), hour_2(_0, _1, ...), rather
    # than interleaving all _0 files, then all _1 files.
    return files


def _infer_shared_time_axis(
    shapes: List[Tuple[int, ...]],
    n_scans_hint: Optional[int],
) -> Optional[int]:
    if not shapes or n_scans_hint is None:
        return None
    rank = len(shapes[0])
    if any(len(shape) != rank for shape in shapes):
        return None

    matches = []
    for axis in range(rank):
        if sum(shape[axis] for shape in shapes) != n_scans_hint:
            continue
        others_match = all(
            all(shape[dim] == shapes[0][dim] for dim in range(rank) if dim != axis)
            for shape in shapes[1:]
        )
        if others_match:
            matches.append(axis)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous time axis for chunked recon with n_scans={n_scans_hint}: matches={matches}"
        )
    return None


def _standardize_recon_part(
    arr: np.ndarray,
    n_scans_hint: Optional[int],
    shared_time_axis: Optional[int] = None,
) -> Tuple[np.ndarray, int, str]:
    if arr.ndim == 5 and arr.shape[-1] == 2:
        t_axis = shared_time_axis
        if t_axis is None:
            t_axis, _ = _infer_time_axis(arr.shape[:-1], n_scans_hint)
        T = arr.shape[:-1][t_axis]
        return _moveaxis_to_penultimate(arr, t_axis), int(T), "real_imag"

    if arr.ndim == 4:
        t_axis = shared_time_axis
        if t_axis is None:
            t_axis, _ = _infer_time_axis(arr.shape, n_scans_hint)
        T = arr.shape[t_axis]
        return _moveaxis_to_last(arr, t_axis), int(T), "dense"

    raise ValueError(f"Unsupported recon shape={arr.shape}, dtype={arr.dtype}")


def _load_standardized_recon(
    npy_path: Any,
    scale: float,
    n_scans_hint: Optional[int],
    time_axis: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    npy_files = _resolve_recon_npy_files(npy_path)
    if len(npy_files) == 1:
        print(f"Loading recon from {npy_files[0]}")
    else:
        print(f"Loading recon sequence from {_path_spec_for_log(npy_path)} ({len(npy_files)} files)")

    raw_shapes: List[Tuple[int, ...]] = []
    shape_kind: Optional[str] = None
    for npy_file in npy_files:
        raw_mmap = np.load(str(npy_file), mmap_mode="r")
        if raw_mmap.ndim == 5 and raw_mmap.shape[-1] == 2:
            current_kind = "real_imag"
            raw_shapes.append(tuple(raw_mmap.shape[:-1]))
        elif raw_mmap.ndim == 4:
            current_kind = "dense"
            raw_shapes.append(tuple(raw_mmap.shape))
        else:
            raise ValueError(f"Unsupported recon shape={raw_mmap.shape}, dtype={raw_mmap.dtype}")

        if shape_kind is None:
            shape_kind = current_kind
        elif current_kind != shape_kind:
            raise ValueError(
                f"Inconsistent recon formats in {npy_path}: got {shape_kind} and {current_kind}"
            )

    shared_time_axis = time_axis
    if shared_time_axis is None:
        shared_time_axis = _infer_shared_time_axis(raw_shapes, n_scans_hint)
    if shared_time_axis is not None and len(npy_files) > 1:
        print(
            f"Using shared time axis={shared_time_axis} for sequence with "
            f"n_scans={n_scans_hint}"
        )

    parts: List[np.ndarray] = []
    total_T = 0
    part_kind: Optional[str] = None

    for npy_file in npy_files:
        raw = np.load(str(npy_file))
        part, part_T, current_kind = _standardize_recon_part(
            raw,
            n_scans_hint,
            shared_time_axis=shared_time_axis,
        )
        if part_kind is None:
            part_kind = current_kind
        elif current_kind != part_kind:
            raise ValueError(
                f"Inconsistent recon formats in {npy_path}: got {part_kind} and {current_kind}"
            )
        parts.append(part)
        total_T += part_T
        print(f"  - {npy_file.name}: raw_shape={raw.shape}, T={part_T}")

    if not parts:
        raise FileNotFoundError(f"No recon arrays could be loaded from {npy_path}")

    concat_axis = -2 if part_kind == "real_imag" else -1
    arr = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=concat_axis)
    arr = arr / float(scale)
    print(f"Combined recon shape={arr.shape}, total_T={total_T}")
    return arr, int(total_T)

def _bg_to_xyzt(bg: np.ndarray, T: int) -> np.ndarray:
    """
    Accept bg as:
    - (X,Y,Z) complex -> broadcast to (X,Y,Z,T)
    - (T,X,Y,Z) or (X,Y,Z,T) complex -> move time to last if needed
    Return (X,Y,Z,T)
    """
    if bg.ndim == 3:
        return bg[..., None] # (X,Y,Z,1) broadcast later
    if bg.ndim == 4:
        # infer which axis is time by matching T
        if bg.shape[-1] == T:
            return bg
        if bg.shape[0] == T:
            return np.moveaxis(bg, 0, -1)
    # ambiguous fallback: assume last is time
        return bg
    raise ValueError(f"bg_complex has invalid ndim={bg.ndim}, shape={bg.shape}")

def load_mag_xyzt_from_npy(
    npy_path: Any,
    scale: float,
    n_scans_hint: Optional[int],
    use_bg_complex: bool,
    bg_complex_npy: Optional[str],
    bg_policy: str,
    time_axis: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Return magnitude in XYZT and T.
    Supports:
      - complex: (T,X,Y,Z) or (X,Y,Z,T)
      - real/imag: (T,X,Y,Z,2) or (X,Y,Z,T,2)
      - magnitude float: (T,X,Y,Z) or (X,Y,Z,T)
      - directory: load xxx_0.npy, xxx_1.npy, ... and concatenate by numeric suffix
    """
    arr, T = _load_standardized_recon(
        npy_path=str(npy_path),
        scale=scale,
        n_scans_hint=n_scans_hint,
        time_axis=time_axis,
    )
    bg_complex = None
    if use_bg_complex and bg_complex_npy is not None:
        bg_complex = np.load(str(bg_complex_npy))

    # real/imag channels
    if arr.ndim == 5 and arr.shape[-1] == 2:
        cplx = arr[..., 0] + 1j * arr[..., 1]
        if bg_policy == "complex_add_then_abs" and bg_complex is not None:
            bg_xyzt = _bg_to_xyzt(bg_complex, T) # (X,Y,Z,T) or (X,Y,Z,1)
            cplx = cplx + bg_xyzt # numpy 会把 (X,Y,Z,1) 广播到 (X,Y,Z,T)
            mag = np.abs(cplx)
        else:
            mag = np.abs(cplx)
            if bg_policy == "abs_add" and bg_complex is not None:
                bgm = np.abs(bg_complex) if np.iscomplexobj(bg_complex) else np.abs(bg_complex.astype(np.float32))
                if bgm.ndim == 4:
                    bg_taxis, _ = _infer_time_axis(bgm.shape, T)
                    bgm = _moveaxis_to_last(bgm, bg_taxis)
                mag = mag + bgm
        return mag.astype(np.float32), int(T)

    # complex
    if np.iscomplexobj(arr):
        cplx = arr  # XYZT
        if bg_policy == "complex_add_then_abs" and bg_complex is not None:
            if bg_complex.ndim == 4:
                bg_taxis, _ = _infer_time_axis(bg_complex.shape, T)
                bg_complex = _moveaxis_to_last(bg_complex, bg_taxis)
            cplx = cplx + bg_complex
            mag = np.abs(cplx)
        else:
            mag = np.abs(cplx)
            if bg_policy == "abs_add" and bg_complex is not None:
                bgm = np.abs(bg_complex) if np.iscomplexobj(bg_complex) else np.abs(bg_complex.astype(np.float32))
                if bgm.ndim == 4:
                    bg_taxis, _ = _infer_time_axis(bgm.shape, T)
                    bgm = _moveaxis_to_last(bgm, bg_taxis)
                mag = mag + bgm
        return mag.astype(np.float32), int(T)

    # magnitude float 4D
    if arr.ndim == 4:
        print(f"Using standardized magnitude input, T={T}")
        mag = arr  # XYZT
        if bg_policy == "abs_add" and bg_complex is not None:
            bgm = np.abs(bg_complex) if np.iscomplexobj(bg_complex) else np.abs(bg_complex.astype(np.float32))
            if bgm.ndim == 4:
                bg_taxis, _ = _infer_time_axis(bgm.shape, T)
                bgm = _moveaxis_to_last(bgm, bg_taxis)
            mag = mag + bgm
        return mag.astype(np.float32), int(T)

    raise ValueError(f"Unsupported recon shape={arr.shape}, dtype={arr.dtype}")


def save_recon_nifti_xyzt(mag_xyzt: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mag_xyzt.astype(np.float32), affine=affine), str(out_path))


def _standardize_rp(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    rp_dst = out_dir / "rp_recon.txt"
    if rp_dst.exists():
        return rp_dst
    rp_candidates = list(out_dir.glob("rp_*.txt"))
    if not rp_candidates:
        raise FileNotFoundError(f"No rp_*.txt in {out_dir}")
    rp_src = sorted(rp_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    if rp_src.resolve() != rp_dst.resolve():
        shutil.copyfile(rp_src, rp_dst)
    return rp_dst


def run_spm_realign(recon_nii: Path, out_dir: Path, cfg: GlobalConfig) -> Path:
    _configure_spm(cfg.matlab_cmd)

    out_dir = Path(out_dir)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # os.environ["TMPDIR"] = str(tmp_dir)
    # os.environ["MCR_CACHE_ROOT"] = str(tmp_dir / "mcr_cache")
    # os.environ["MATLAB_PREFDIR"] = str(tmp_dir / "matlab_pref")
    

    cwd = os.getcwd()
    try:
        os.chdir(str(out_dir))
        realign = spm.Realign()
        realign.inputs.in_files = [str(recon_nii)]
        realign.inputs.register_to_mean = False
        realign.inputs.jobtype = cfg.spm_jobtype  # "estimate" recommended
        #realign.terminal_output = "file"
        res = realign.run()

        (out_dir / "spm_stdout.txt").write_text(res.runtime.stdout or "")
        (out_dir / "spm_stderr.txt").write_text(res.runtime.stderr or "")

        return _standardize_rp(out_dir)
    finally:
        os.chdir(cwd)


def _get_bg_img(out_dir: Path, cfg: GlobalConfig, spec: ReconSpec) -> nib.Nifti1Image:
    if spec.bg_mode == "mean_func":
        img = nib.load(str(out_dir / "recon.nii"))
        dat = img.get_fdata()
        last3d = dat[..., -1].astype(np.float32)
        return nib.Nifti1Image(last3d, affine=img.affine)

    # t1
    t1_path = spec.bg_t1_path if spec.bg_t1_path is not None else cfg.default_t1_path
    return nib.load(str(t1_path))


def run_glm_and_save(out_dir: Path, cfg: GlobalConfig, tr: float, T: int) -> Tuple[float, int, float]:
    out_dir = Path(out_dir)
    recon_nii = out_dir / "recon.nii"
    rp_txt = out_dir / "rp_recon.txt"

    r_img = nib.load(str(recon_nii))
    motion = np.loadtxt(str(rp_txt))

    z_map, dm, glm, events = run_first_level_single_session(
        r_img,
        motion_params=motion,
        n_scans=int(T),
        tr=tr,
    )

    nib.save(z_map, str(out_dir / "z_map_glob.nii.gz"))

    # 保存 design matrix（你说多数情况只看 activation，但 dm 需要保留）
    if cfg.save_design_matrix:
        try:
            if hasattr(dm, "to_csv"):
                dm.to_csv(out_dir / "design_matrix.tsv", sep="\t", index=False)
        except Exception:
            pass
        try:
            ax = plot_design_matrix(dm)
            fig = ax.figure
            fig.set_size_inches(12, 3)
            fig.tight_layout()
            fig.savefig(str(out_dir / "design_matrix.png"), dpi=150)
            fig.clf()
        except Exception as e:
            (out_dir / "WARN_design_matrix.txt").write_text(repr(e))

    # events
    try:
        (out_dir / "events.json").write_text(json.dumps(events, indent=2, default=str))
    except Exception:
        (out_dir / "events.txt").write_text(str(events))

    # threshold (default settings)
    stat_img, thr = threshold_stats_img(
        z_map,
        alpha=float(cfg.alpha_fpr_default),
        height_control="fpr",
        cluster_threshold=int(cfg.cluster_th_default),
    )
    nib.save(stat_img, str(out_dir / "z_map_thr.nii.gz"))

    stat_data = stat_img.get_fdata()
    n_supra = int(np.sum(np.isfinite(stat_data) & (stat_data != 0)))
    max_z = float(np.nanmax(stat_data)) if np.any(np.isfinite(stat_data)) else float("nan")

    return float(thr), n_supra, max_z


def make_overlay_png(
    out_dir: Path,
    cfg: GlobalConfig,
    spec: ReconSpec,
    alpha: float,
    cluster_th: int,
    cut_coords: Tuple[float, float, float],
    out_png: Optional[Path] = None,
    layout: str = "ortho",
) -> Path:
    """
    只用于可视化：读取 z_map_glob，按 alpha/cluster 做阈值+cluster，然后 resample 到 bg，
    保存 overlay PNG，返回 PNG 路径。
    """
    out_dir = Path(out_dir)
    z_map = nib.load(str(out_dir / "z_map_glob.nii.gz"))
    bg_img = _get_bg_img(out_dir, cfg, spec)
    bg_abs = np.abs(bg_img.get_fdata())
    bg_abs = np.nan_to_num(bg_abs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    bg_plot_img = nib.Nifti1Image(bg_abs, affine=bg_img.affine, header=bg_img.header)

    stat_img, thr = threshold_stats_img(
        z_map,
        alpha=float(alpha),
        height_control="fpr",
        cluster_threshold=int(cluster_th),
    )

    stat_res = resample_to_img(
        stat_img, bg_img,
        interpolation="continuous",
        force_resample=bool(cfg.force_resample),
        copy_header=bool(cfg.copy_header),
    )

    if out_png is None:
        out_png = out_dir / f"QC_overlay_alpha{alpha:.1e}_cl{int(cluster_th)}.png"

    if layout == "ortho":
        disp = plot_stat_map(
            stat_res,
            threshold=float(thr),
            bg_img=bg_plot_img,
            draw_cross=False,
            cut_coords=list(cut_coords),
            dim=False,
            vmin=-18,
            vmax=18,
        )
        colorbar_ax = getattr(disp, "_colorbar_ax", None)
        colorbar_axes = [colorbar_ax] if colorbar_ax is not None else []
        if not colorbar_axes:
            colorbar_axes = [
                ax for ax in disp.frame_axes.figure.axes
                if ax.get_position().width < 0.08 and ax.get_position().height > 0.25
            ]
        for ax in colorbar_axes:
            fig = ax.figure
            bbox = ax.get_position()
            ax.set_position([bbox.x0 - 0.045, bbox.y0, bbox.width, bbox.height])
            bbox = ax.get_position()
            pad_x, pad_y = 0.012, 0.015
            fig.patches.append(Rectangle(
                (bbox.x0 - pad_x, bbox.y0 - pad_y),
                bbox.width + 0.085,
                bbox.height + 2 * pad_y,
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="none",
                zorder=-1,
            ))
            ax.set_facecolor("white")
            ax.patch.set_alpha(1.0)
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
            ticks = np.linspace(-18, 18, 5)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{tick:g}" for tick in ticks])
            ax.tick_params(
                axis="y",
                colors="black",
                labelleft=False,
                labelright=True,
                left=False,
                right=True,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
            for tick_label in ax.get_yticklabels():
                tick_label.set_visible(True)
                tick_label.set_color("black")

        disp.savefig(str(out_png), dpi=150)
        disp.close()
        return Path(out_png)

    if layout != "mosaic":
        raise ValueError(f"layout must be 'ortho' or 'mosaic', got {layout!r}")

    # Mosaic QC layout: a large axial view on the left, with sagittal and
    # coronal views stacked on the right.  cut_coords follows Nilearn's
    # conventional (x, y, z) order.
    bg_max = float(np.max(bg_abs))
    r1_298_bg_dir = Path(
        "/volatile/Caini/mnt/topaze/scratch/results_3T_invivo/"
        "nifti_1219/R1_298_cg"
    ).resolve()
    orc_08_bg_dirs = {
        Path(
            "/volatile/Caini/mnt/topaze/scratch/results_3T_invivo/"
            f"nifti_1219/{label}"
        ).resolve()
        for label in ("R1_298_cg_orc", "R1_288_cg_orc")
    }
    if out_dir.resolve() == r1_298_bg_dir:
        bg_vmin = 0.05 * bg_max
        bg_vmax = bg_max
    elif out_dir.resolve() in orc_08_bg_dirs:
        bg_vmin = 0.05 * bg_max
        bg_vmax = 0.80 * bg_max
    else:
        bg_vmin = 0.05 * bg_max
        bg_vmax = 0.70 * bg_max
    if bg_vmax <= bg_vmin:
        bg_vmax = bg_vmin + 1e-6
    bg_windowed = np.clip(bg_abs, bg_vmin, bg_vmax)
    bg_mosaic_img = nib.Nifti1Image(
        bg_windowed.astype(np.float32),
        affine=bg_img.affine,
        header=bg_img.header,
    )

    x_coord, y_coord, z_coord = (float(coord) for coord in cut_coords)
    fig = plt.figure(figsize=(8.4, 5.4), facecolor="black")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.35, 1.0, 0.055),
        height_ratios=(1.0, 1.0),
        left=0.01,
        right=0.94,
        bottom=0.02,
        top=0.98,
        wspace=0.0,
        hspace=0.0,
    )
    view_axes = {
        "z": (fig.add_subplot(grid[:, 0]), z_coord),
        "x": (fig.add_subplot(grid[0, 1]), x_coord),
        "y": (fig.add_subplot(grid[1, 1]), y_coord),
    }

    displays = []
    for display_mode, (ax, coord) in view_axes.items():
        displays.append(plot_stat_map(
            stat_res,
            threshold=float(thr),
            bg_img=bg_mosaic_img,
            display_mode=display_mode,
            cut_coords=[coord],
            axes=ax,
            figure=fig,
            colorbar=False,
            annotate=False,
            draw_cross=False,
            dim=False,
            cmap="RdBu_r",
            vmin=-18,
            vmax=18,
        ))

    # One shared colorbar keeps all three panels the same size and scale.
    colorbar_ax = fig.add_subplot(grid[:, 2])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-18, vmax=18), cmap="RdBu_r"),
        cax=colorbar_ax,
        ticks=np.linspace(-18, 18, 5),
    )
    colorbar.ax.set_yticklabels([f"{tick:g}" for tick in np.linspace(-18, 18, 5)])
    colorbar.ax.tick_params(axis="y", colors="white", labelsize=8)
    colorbar.outline.set_edgecolor("white")

    fig.savefig(str(out_png), dpi=150, facecolor="black")
    plt.close(fig)
    return Path(out_png)


def plot_bg_slices_png(
    out_dir: Path,
    cfg: GlobalConfig,
    spec: ReconSpec,
    plane: str = "axial",
    title: Optional[str] = None,
    out_png: Optional[Path] = None,
    n_cols: int = 8,
    dpi: int = 150,
) -> Path:
    """
    Plot all background slices in one grid and save as PNG.

    plane:
      - "axial": plot z slices, data[:, :, k]
      - "sagittal": plot x slices, data[k, :, :]
      - "coronal": plot y slices, data[:, k, :]
    """
    out_dir = Path(out_dir)
    plane = plane.lower()
    if plane not in {"axial", "sagittal", "coronal"}:
        raise ValueError(f"plane must be 'axial', 'sagittal', or 'coronal', got {plane!r}")

    bg_img = _get_bg_img(out_dir, cfg, spec)
    bg = np.abs(bg_img.get_fdata())
    bg = np.nan_to_num(bg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if plane == "axial":
        n_slices = bg.shape[2]
        get_slice = lambda i: np.rot90(bg[:, :, i])
    elif plane == "sagittal":
        n_slices = bg.shape[0]
        get_slice = lambda i: np.rot90(bg[i, :, :])
    else:
        n_slices = bg.shape[1]
        get_slice = lambda i: np.rot90(bg[:, i, :])

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_slices / n_cols))
    bg_vmin = float(np.min(bg))
    bg_vmax = float(np.max(bg))
    if bg_vmax <= bg_vmin:
        bg_vmax = bg_vmin + 1e-6

    fig_w = max(8.0, 1.6 * n_cols)
    fig_h = max(3.0, 1.6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.ravel()

    for i in range(n_slices):
        ax = axes[i]
        ax.imshow(get_slice(i), cmap="gray", vmin=bg_vmin, vmax=bg_vmax, origin="lower")
        ax.axis("off")

    for ax in axes[n_slices:]:
        ax.axis("off")

    if title is None:
        label = spec.label if spec.label is not None else infer_label_from_recon_path_spec(spec.path)
        title = f"{label} background slices ({plane}, bg_mode={spec.bg_mode})"
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.01, hspace=0.01)

    if out_png is None:
        out_png = out_dir / f"QC_bg_{plane}_slices.png"
    fig.savefig(str(out_png), dpi=int(dpi))
    plt.close(fig)
    return Path(out_png)


def process_one(spec: ReconSpec, cfg: GlobalConfig, affine: np.ndarray) -> OneReconResult:
    rp = spec.path
    rp_log = _path_spec_for_log(rp)
    label = spec.label if spec.label is not None else infer_label_from_recon_path_spec(rp)
    father_out = Path(cfg.father_out)
    out_dir = father_out / label
    did_reset = _maybe_reset_out_dir(out_dir, father_out, spec.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _logger(out_dir)
    if did_reset:
        log("OVERWRITE: cleared previous outputs for this label.")

    done_flag = out_dir / "_DONE.txt"
    if done_flag.exists():
        log("SKIP: _DONE.txt exists.")
        return OneReconResult(label=label, recon_path=rp_log, out_dir=str(out_dir), status="skipped")

    try:
        log(f"START: {label}")
        log(f"Recon path: {rp_log}")

        # Step 1: npy -> recon.nii (XYZT)
        log("Step 1/3: npy -> recon.nii (magnitude, XYZT in NIfTI)")
        mag_xyzt, T = load_mag_xyzt_from_npy(
            npy_path=rp,
            scale=spec.scale,
            n_scans_hint=spec.n_scans,
            time_axis=spec.recon_time_axis,
            use_bg_complex=spec.use_bg_complex,
            bg_complex_npy=spec.bg_complex_npy,
            bg_policy=spec.bg_policy,
        )
        save_recon_nifti_xyzt(mag_xyzt, affine, out_dir / "recon.nii")
        log(f"Saved recon.nii shape={mag_xyzt.shape}, T={T}")

        # Step 2: SPM realign -> rp_recon.txt
        log("Step 2/3: SPM Realign -> rp_recon.txt")
        rp_txt = run_spm_realign(out_dir / "recon.nii", out_dir, cfg)
        log(f"Motion params: {rp_txt}")

        # Step 3: GLM + save z map + dm
        log("Step 3/3: GLM -> z_map_glob + threshold(default) + dm saved")
        thr, n_supra, max_z = run_glm_and_save(out_dir, cfg, float(spec.tr), T)
        log(f"GLM done thr={thr:.4f}, n_supra={n_supra}, max_z={max_z:.3f}")

        report = {
            "label": label,
            "recon_path": rp_log,
            "T": int(T),
            "TR": float(cfg.tr),
            "scale": float(spec.scale),
            "recon_time_axis": spec.recon_time_axis,
            "overwrite": bool(spec.overwrite),
            "spm_jobtype": cfg.spm_jobtype,
            "alpha_fpr_default": float(cfg.alpha_fpr_default),
            "cluster_th_default": int(cfg.cluster_th_default),
            "bg_mode": spec.bg_mode,
            "bg_t1_path": spec.bg_t1_path,
            "bg_policy": spec.bg_policy,
            "use_bg_complex": bool(spec.use_bg_complex),
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))

        done_flag.write_text("OK\n")
        log("DONE.")
        return OneReconResult(
            label=label, recon_path=rp_log, out_dir=str(out_dir),
            status="ok", T=int(T), threshold=float(thr),
            n_supra=int(n_supra), max_z=float(max_z)
        )

    except Exception:
        tb = traceback.format_exc()
        (out_dir / "_ERROR.txt").write_text(tb)
        log("FAILED. See _ERROR.txt")
        log(tb)
        return OneReconResult(label=label, recon_path=rp_log, out_dir=str(out_dir), status="failed", error="see _ERROR.txt")


def run_batch(specs: List[ReconSpec], cfg: GlobalConfig, affine: np.ndarray) -> List[OneReconResult]:
    specs = list(specs)
    father = Path(cfg.father_out)
    father.mkdir(parents=True, exist_ok=True)

    # Parallel across reconstructions
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=int(cfg.n_jobs), backend="loky", verbose=10)(
            delayed(process_one)(spec, cfg, affine) for spec in specs
        )
    except Exception:
        results = [process_one(spec, cfg, affine) for spec in specs]

    # summary
    import csv
    summary_csv = father / "batch_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "status", "T", "thr_default", "n_supra", "max_z", "out_dir", "recon_path", "error"])
        for r in results:
            w.writerow([r.label, r.status, r.T, r.threshold, r.n_supra, r.max_z, r.out_dir, r.recon_path, r.error])

    return results



def read_rp_txt(rp_path: str | Path, rot_unit: str = "rad") -> np.ndarray:
    """
    读取 SPM rp_*.txt, 返回 shape (T, 6)
    rot_unit:
      - "rad": 保持原始弧度
      - "deg": 把旋转(4-6列)转换为角度
    """
    rp = np.loadtxt(str(rp_path))
    if rp.ndim != 2 or rp.shape[1] < 6:
        raise ValueError(f"rp file shape invalid: {rp.shape}")
    rp = rp[:, :6].astype(np.float64)

    if rot_unit == "deg":
        rp[:, 3:6] = rp[:, 3:6] * (180.0 / np.pi)
    elif rot_unit != "rad":
        raise ValueError("rot_unit must be 'rad' or 'deg'")
    return rp


def collect_rp_from_methods(
    father_out: str | Path,
    labels: list[str] | None = None,
    rp_name: str = "rp_recon.txt",
    rot_unit: str = "rad",
) -> dict[str, np.ndarray]:
    """
    从 father_out/<label>/rp_recon.txt 收集所有方法的 rp，返回 dict[label] = (T,6)
    labels:
      - None: 自动扫描 father_out 下所有包含 rp_name 的子目录
      - list: 指定方法列表（更可控）
    """
    father = Path(father_out)
    if labels is None:
        labels = sorted([p.name for p in father.iterdir() if p.is_dir() and (p / rp_name).exists()])

    out = {}
    for lb in labels:
        rp_path = father / lb / rp_name
        if not rp_path.exists():
            continue
        out[lb] = read_rp_txt(rp_path, rot_unit=rot_unit)
    if len(out) == 0:
        raise FileNotFoundError(f"No {rp_name} found under {father}")
    return out


def framewise_displacement(rp: np.ndarray, rot_in: str = "rad", radius_mm: float = 50.0) -> np.ndarray:
    """
    计算一个常用的 FD（Power-like）指标，基于帧间差分：
      FD[t] = sum(|Δtrans|) + radius * sum(|Δrot|)
    rp: (T,6)  [mm, mm, mm, rad, rad, rad] if rot_in='rad'
    rot_in: 'rad' 或 'deg'（若是 deg，会先转 rad）
    """
    x = rp.copy().astype(np.float64)
    if rot_in == "deg":
        x[:, 3:6] = x[:, 3:6] * (np.pi / 180.0)
    elif rot_in != "rad":
        raise ValueError("rot_in must be 'rad' or 'deg'")

    dx = np.diff(x, axis=0, prepend=x[[0], :])
    fd = np.sum(np.abs(dx[:, 0:3]), axis=1) + radius_mm * np.sum(np.abs(dx[:, 3:6]), axis=1)
    return fd


def plot_motion_designmatrix_imshow(
    rp_dict: dict[str, np.ndarray],
    standardize: str = "per_param_global",
    title: str = "Motion parameters (stacked methods)  [rows=methods*time, cols=6]",
    param_names: list[str] | None = None,
    figsize=(8, 8),
):
    """
    把多个方法的 rp 纵向堆叠成一个大矩阵，然后用 imshow 画成“一个 design matrix 的样子”。
    rp_dict: {label: (T,6)}

    standardize:
      - "none": 不标准化（注意平移mm和旋转rad量纲不同，imshow对比不直观）
      - "per_method_param": 每个方法内部每一列做 z-score（更适合看结构/时间模式）
      - "per_param_global": 对每一列在所有方法/时间上做 z-score（最推荐用于跨方法比较）
    """
    labels = list(rp_dict.keys())
    mats = [rp_dict[k] for k in labels]
    Ts = [m.shape[0] for m in mats]
    if len(set([m.shape[1] for m in mats])) != 1:
        raise ValueError("All rp must have same number of columns")
    C = mats[0].shape[1]

    big = np.vstack(mats)  # (sumT, 6)

    if standardize == "per_method_param":
        big2 = []
        start = 0
        for T in Ts:
            seg = big[start:start+T]
            mu = seg.mean(axis=0, keepdims=True)
            sd = seg.std(axis=0, keepdims=True) + 1e-12
            big2.append((seg - mu) / sd)
            start += T
        big = np.vstack(big2)
    elif standardize == "per_param_global":
        mu = big.mean(axis=0, keepdims=True)
        sd = big.std(axis=0, keepdims=True) + 1e-12
        big = (big - mu) / sd
    elif standardize != "none":
        raise ValueError("standardize must be 'none'|'per_method_param'|'per_param_global'")

    if param_names is None:
        param_names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(big, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("motion parameters (6)")
    ax.set_xticks(range(C))
    ax.set_xticklabels(param_names, rotation=45, ha="right")

    # y ticks at method centers + horizontal separators
    cum = np.cumsum([0] + Ts)
    centers = [(cum[i] + cum[i+1]) / 2 for i in range(len(Ts))]
    ax.set_yticks(centers)
    ax.set_yticklabels(labels)

    for y in cum[1:-1]:
        ax.axhline(y - 0.5, linewidth=1)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def motion_diff_mean_var_matrix(rp_dict: dict[str, np.ndarray], use_abs_mean: bool = True):
    """
    对每个方法计算 Δrp 的统计：
      mean_df: mean(|Δ|)（默认）或 mean(Δ)
      var_df : var(Δ)
    rows=methods, cols=6 parameters
    """
    labels = list(rp_dict.keys())
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

    mean_rows, var_rows = [], []
    for lb in labels:
        rp = rp_dict[lb].astype(np.float64)
        drp = np.diff(rp, axis=0, prepend=rp[[0], :])  # (T,6)
        if use_abs_mean:
            mean_rows.append(np.mean(np.abs(drp), axis=0))
        else:
            mean_rows.append(np.mean(drp, axis=0))
        var_rows.append(np.var(drp, axis=0))

    mean_df = pd.DataFrame(mean_rows, index=labels, columns=cols)
    var_df  = pd.DataFrame(var_rows,  index=labels, columns=cols)
    return mean_df, var_df


def plot_mean_var_annotated_matrix(
    mean_df: pd.DataFrame,
    var_df: pd.DataFrame,
    color_by: str = "var",
    title: str = "Motion summary (each cell: mean / var)",
    figsize=(10, 6),
    fmt_mean: str = "{:.3g}",
    fmt_var: str = "{:.3g}",
):
    """
    画一个矩阵：行=方法，列=6参数，每个格子写两行文字 mean 和 var。
    背景颜色可以选择用 var 或 mean 的绝对值。
    """
    methods = list(mean_df.index)
    params = list(mean_df.columns)
    M, P = len(methods), len(params)

    if color_by == "var":
        base = var_df.values
    elif color_by == "abs_mean":
        base = np.abs(mean_df.values)
    else:
        raise ValueError("color_by must be 'var' or 'abs_mean'")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(base, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(P))
    ax.set_xticklabels(params, rotation=45, ha="right")
    ax.set_yticks(range(M))
    ax.set_yticklabels(methods)

    for i in range(M):
        for j in range(P):
            mu = mean_df.values[i, j]
            va = var_df.values[i, j]
            txt = f"μ={fmt_mean.format(mu)}\nσ²={fmt_var.format(va)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def plot_fd_timeseries(
    rp_dict: dict[str, np.ndarray],
    rot_in: str = "rad",
    radius_mm: float = 50.0,
    title: str = "Framewise displacement (FD)",
    figsize=(10, 5),
):
    """
    画多方法的 FD 时间序列（同一张图）。
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for lb, rp in rp_dict.items():
        fd = framewise_displacement(rp, rot_in=rot_in, radius_mm=radius_mm)
        ax.plot(fd, label=lb)
    ax.set_title(title)
    ax.set_xlabel("time (frame)")
    ax.set_ylabel("FD (mm)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax
