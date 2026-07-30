from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

from nilearn.glm.thresholding import threshold_stats_img
from nilearn.image import resample_to_img, math_img


from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    VPacker,
    HPacker,
    TextArea,
    AnchoredOffsetbox,
    DrawingArea,
    PaddedBox,
)
from matplotlib.legend import Legend
from sklearn.metrics import auc
from snake.toolkit.analysis.stats import bacc
from scipy.stats import norm, wilcoxon
from statsmodels.stats.multitest import multipletests
# ----------------------------
# 0) 配置与数据结构
# ----------------------------

RUN_CFG_DEFAULT = {
    "R1": dict(n_scans=120, tr=2.0),
    "R2": dict(n_scans=240, tr=1.0),
    "R4": dict(n_scans=480, tr=0.5),
    "R8": dict(n_scans=960, tr=0.25),
}

@dataclass
class CasePack:
    z_map: nib.Nifti1Image
    dm: object
    glm: object
    events: object
    img: nib.Nifti1Image
    motion: np.ndarray
    paths: dict
    meta: dict


@dataclass
class BaseResult:
    base: Path
    date_tag: str
    results_by_folder: dict[str, CasePack]
    union_mask_bool: np.ndarray  # boolean mask on ref_img grid
    overlap_mask_bool: np.ndarray
    anchor_info: dict            # info about which two R1 used, etc.


# ----------------------------
# 1) 小工具：解析目录名、阈值成 mask、指标
# ----------------------------

def infer_date_tag_from_base(base: Path) -> str:
    """
    例如 base.name = 'nifti_0210' -> '0210'
    如果不符合，也就原样返回 base.name
    """
    m = re.search(r"nifti[_\-]?(\d+)$", base.name, flags=re.IGNORECASE)
    return m.group(1) if m else base.name


def parse_run_and_method(folder_name: str):
    name = folder_name

    # 1) run: 必须以 R1/R2/R4/R8 开头，后面要么是 '_' 要么直接结束
    m_run = re.match(r"^(R1|R2|R4|R8)(?:_|$)", name, flags=re.IGNORECASE)
    if not m_run:
        raise ValueError(f"Folder name does not start with R1/R2/R4/R8: {folder_name}")
    run = m_run.group(1).upper()

    # 2) method: 只要包含 'global' 子串就算 global，否则 cold
    method = "global" if re.search(r"global", name, flags=re.IGNORECASE) else "cold"

    # 3) idx: 抓 R1_66_... 里的 66（允许后面是 '_' 或结束）
    m_idx = re.match(r"^(R1|R2|R4|R8)_(\d+)(?:_|$)", name, flags=re.IGNORECASE)
    idx = int(m_idx.group(2)) if m_idx else None

    return run, method, idx


def zmap_to_binary_mask(
    z_map: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
    alpha: float = 0.001,
    height_control: str = "fpr",
    cluster_threshold: int = 50,
    two_sided: bool = True,
) -> tuple[nib.Nifti1Image, np.ndarray, float]:
    """
    z_map -> threshold+cluster -> binary -> resample to ref_img (nearest)
    return: (mask_img_resampled, mask_bool, thr)
    """
    stat_img, thr = threshold_stats_img(
        z_map,
        alpha=alpha,
        height_control=height_control,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )
    mask_img = math_img("img != 0", img=stat_img)
    mask_img_r = resample_to_img(mask_img, ref_img, interpolation="nearest")
    mask_bool = np.nan_to_num(mask_img_r.get_fdata()) != 0
    return mask_img_r, mask_bool, float(thr)


def compute_dice_fbeta(
    gt_bool: np.ndarray,
    pred_bool: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, float]:
    """
    gt_bool: union mask (groundtruth proxy)
    pred_bool: prediction mask for a case
    """
    gt_n = int(np.count_nonzero(gt_bool))
    pred_n = int(np.count_nonzero(pred_bool))
    overlap_n = int(np.count_nonzero(gt_bool & pred_bool))

    denom = gt_n + pred_n
    dice = (2.0 * overlap_n / denom) if denom > 0 else 0.0

    P = (overlap_n / pred_n) if pred_n > 0 else 0.0
    R = (overlap_n / gt_n) if gt_n > 0 else 0.0
    denom_f = (beta**2 * P + R)
    fbeta = ((1 + beta**2) * P * R / denom_f) if denom_f > 0 else 0.0
    return float(dice), float(fbeta)


# ----------------------------
# 2) 核心：处理一个 BASE
# ----------------------------

def build_results_by_folder_for_base(
    base: Path,
    run_first_level_single_session,
    run_cfg: dict[str, dict] = RUN_CFG_DEFAULT,
    nifti_name: str = "recon.nii",
    motion_name: str = "rp_recon.txt",
    zmap_cache_name: str = "z_map_glob.nii.gz",
    force_recompute: bool = False,
    folder_filter=None,
    skip_missing: bool = True,
) -> dict[str, CasePack]:
    """
    扫描 base 下子文件夹，逐个跑 GLM，返回扁平 results_by_folder。
    """
    if not base.is_dir():
        raise NotADirectoryError(f"BASE not found: {base}")

    results_by_folder: dict[str, CasePack] = {}

    for d in sorted([p for p in base.iterdir() if p.is_dir()]):
        try:
            run, method, idx = parse_run_and_method(d.name)
        except ValueError:
            continue  # 不符合命名规则的文件夹直接跳过

        if folder_filter is not None and not folder_filter(d.name):
            continue

        cfg = run_cfg[run]
        nifti_path = d / nifti_name
        motion_path = d / motion_name
        zmap_path = d / zmap_cache_name
        
        if not nifti_path.is_file():
            if skip_missing:
                print(f"[skip] Missing NIfTI in {d.name}: {nifti_path}")
                continue
            raise FileNotFoundError(f"Missing NIfTI in {d.name}: {nifti_path}")
        if not motion_path.is_file():
            if skip_missing:
                print(f"[skip] Missing motion in {d.name}: {motion_path}")
                continue
            raise FileNotFoundError(f"Missing motion in {d.name}: {motion_path}")

        img = nib.load(str(nifti_path))
        motion = np.loadtxt(str(motion_path))
        if (not force_recompute) and zmap_path.is_file():
            z_map = nib.load(str(zmap_path))
            dm = glm = events = res = None  # 缓存模式下没有这些对象（你如果需要也可以另存）
        else:
            z_map, dm, glm, events = run_first_level_single_session(
                img,
                motion_params=motion,
                n_scans=cfg["n_scans"],
                tr=cfg["tr"],
            )
            nib.save(z_map, str(zmap_path))

        pack = CasePack(
            z_map=z_map, dm=dm, glm=glm, events=events,
            img=img, motion=motion,
            paths=dict(folder=d, nifti=nifti_path, motion=motion_path),
            meta=dict(run=run, idx=idx, method=method, folder=d.name),
        )
        results_by_folder[d.name] = pack

    if len(results_by_folder) == 0:
        raise RuntimeError(f"No valid subfolders found under {base}")
    return results_by_folder


def make_union_from_two_r1_cases(
    results_by_folder: dict[str, CasePack],
    ref_img: nib.Nifti1Image,
    r1_folder_a: str,
    r1_folder_b: str,
    thresh_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    你指定两个 R1 文件夹名，用它们的 z_map 做 mask0_a/mask0_b，
    返回 overlap/union 以及一些信息。
    """
    if thresh_kwargs is None:
        thresh_kwargs = dict(alpha=0.001, height_control="fpr", cluster_threshold=50, two_sided=True)

    if r1_folder_a not in results_by_folder:
        raise KeyError(f"Anchor folder not found: {r1_folder_a}")
    if r1_folder_b not in results_by_folder:
        raise KeyError(f"Anchor folder not found: {r1_folder_b}")

    z_a = results_by_folder[r1_folder_a].z_map
    z_b = results_by_folder[r1_folder_b].z_map

    _, mask_a, thr_a = zmap_to_binary_mask(z_a, ref_img=ref_img, **thresh_kwargs)
    _, mask_b, thr_b = zmap_to_binary_mask(z_b, ref_img=ref_img, **thresh_kwargs)

    overlap = mask_a & mask_b
    union = mask_a | mask_b

    info = dict(
        anchor_a=r1_folder_a, anchor_b=r1_folder_b,
        thr_a=thr_a, thr_b=thr_b,
        n_a=int(np.count_nonzero(mask_a)),
        n_b=int(np.count_nonzero(mask_b)),
        n_overlap=int(np.count_nonzero(overlap)),
        n_union=int(np.count_nonzero(union)),
    )
    return overlap, union, info


def score_all_cases_against_union(
    results_by_folder: dict[str, CasePack],
    ref_img: nib.Nifti1Image,
    union_mask_bool: np.ndarray,
    beta: float = 2.0,
    thresh_kwargs: dict | None = None,
) -> list[dict]:
    """
    对每个 folder 生成 pred mask，然后计算 dice/f2。
    输出 list[record]，每个 record 带 run/method/idx/date等信息。
    """
    if thresh_kwargs is None:
        thresh_kwargs = dict(alpha=0.001, height_control="fpr", cluster_threshold=50, two_sided=True)

    records = []
    for folder, pack in results_by_folder.items():
        _, pred_bool, thr = zmap_to_binary_mask(pack.z_map, ref_img=ref_img, **thresh_kwargs)
        dice, f2 = compute_dice_fbeta(union_mask_bool, pred_bool, beta=beta)

        run = pack.meta["run"]  # "R1" etc
        method = pack.meta["method"]  # "cold"/"global"
        idx = pack.meta["idx"]

        records.append(dict(
            folder=folder,
            run=run,
            R=int(run[1:]),
            method=method,
            idx=idx,
            thr=thr,
            pred_n=int(np.count_nonzero(pred_bool)),
            union_n=int(np.count_nonzero(union_mask_bool)),
            dice=dice,
            f2=f2,
        ))
    return records


def process_one_base(
    base: Path,
    ref_img: nib.Nifti1Image,
    run_first_level_single_session,
    r1_folder_a: str,
    r1_folder_b: str,
    run_cfg: dict[str, dict] = RUN_CFG_DEFAULT,
    beta: float = 2.0,
    thresh_kwargs: dict | None = None,
    folder_filter=None,
) -> tuple[BaseResult, list[dict]]:
    """
    一键处理一个 BASE：扫目录 -> GLM -> union -> 全部打分
    """
    date_tag = infer_date_tag_from_base(base)
    results_by_folder = build_results_by_folder_for_base(
        base=base,
        run_first_level_single_session=run_first_level_single_session,
        run_cfg=run_cfg,
        folder_filter=folder_filter,
    )

    overlap, union, anchor_info = make_union_from_two_r1_cases(
        results_by_folder=results_by_folder,
        ref_img=ref_img,
        r1_folder_a=r1_folder_a,
        r1_folder_b=r1_folder_b,
        thresh_kwargs=thresh_kwargs,
    )

    records = score_all_cases_against_union(
        results_by_folder=results_by_folder,
        ref_img=ref_img,
        union_mask_bool=union,
        beta=beta,
        thresh_kwargs=thresh_kwargs,
    )

    base_res = BaseResult(
        base=base,
        date_tag=date_tag,
        results_by_folder=results_by_folder,
        union_mask_bool=union,
        overlap_mask_bool=overlap,
        anchor_info=anchor_info,
    )
    return base_res, records


# ----------------------------
# 3) 多 BASE 聚合与绘图

def load_compute_spiral_f2_details(
    csv_path: str | Path,
    score_key: str = "f2",
) -> list[dict]:
    """
    读取 compute_spiral_f2.py 输出的 detail csv，并转成统一 records。
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Simulation detail csv not found: {csv_path}")

    records: list[dict] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"phantom_id", "recon", "R", score_key}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Simulation detail csv missing columns {missing}: {csv_path}"
            )

        for row in reader:
            recon = str(row["recon"]).strip().lower()
            if recon not in {"r1", "cold", "global"}:
                continue

            try:
                score_val = float(row[score_key])
            except (TypeError, ValueError):
                score_val = float("nan")

            rec = dict(
                source="simulation",
                method=recon,
                recon=recon,
                R=int(row["R"]),
                phantom_id=int(row["phantom_id"]),
                run_dir=row.get("run_dir", ""),
                f2=score_val if score_key == "f2" else float(row.get("f2", "nan")),
            )

            for key in ("tp", "fp", "fn", "tn"):
                val = row.get(key, "")
                rec[key] = int(val) if val not in ("", None) else None

            if score_key != "f2":
                rec[score_key] = score_val

            records.append(rec)

    if len(records) == 0:
        raise ValueError(f"No usable simulation rows found in {csv_path}")
    return records


def _compute_mean_error_by_method_R(
    records: list[dict],
    score_key: str,
    method_order: tuple[str, ...] | list[str],
    error_mode: str = "sem",
    bridge_missing_R1_for_global: bool = True,
    bridge_source_method: str = "cold",
) -> dict[str, dict[int, dict[str, float]]]:
    if error_mode not in {"std", "sem"}:
        raise ValueError("error_mode must be one of {'std', 'sem'}")

    vals_by_method_R: dict[str, dict[int, list[float]]] = {
        method: {} for method in method_order
    }

    for rec in records:
        method = rec.get("method")
        if method not in vals_by_method_R or score_key not in rec:
            continue

        value = float(rec[score_key])
        if not np.isfinite(value):
            continue

        R = int(rec["R"])
        vals_by_method_R[method].setdefault(R, []).append(value)

    stats_by_method_R: dict[str, dict[int, dict[str, float]]] = {
        method: {} for method in method_order
    }
    for method in method_order:
        for R, vals in vals_by_method_R[method].items():
            arr = np.asarray(vals, dtype=float)
            mu = float(np.mean(arr))
            if arr.size <= 1:
                err = 0.0
            elif error_mode == "sem":
                err = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            else:
                err = float(np.std(arr, ddof=1))

            stats_by_method_R[method][R] = {
                "mean": mu,
                "error": err,
                "n": int(arr.size),
            }

    if (
        bridge_missing_R1_for_global
        and "global" in stats_by_method_R
        and 1 not in stats_by_method_R["global"]
        and bridge_source_method in stats_by_method_R
        and 1 in stats_by_method_R[bridge_source_method]
    ):
        stats_by_method_R["global"][1] = {
            **stats_by_method_R[bridge_source_method][1],
            "bridged_from": bridge_source_method,
        }

    return stats_by_method_R


VISIT_LABEL_BY_DATE = {
    "1219": "V#1",
    "0210": "V#2",
    "0224": "V#4",
    "0306": "V#5",
}
VISIT_MARKER_BY_LABEL = {
    "V#1": "o",
    "V#2": "s",
    "V#4": "^",
    "V#5": "D",
}


def _visit_label_for_record(rec: dict) -> str:
    date_tag = str(rec.get("date_tag", ""))
    return VISIT_LABEL_BY_DATE.get(date_tag, date_tag or "visit")


def _p_to_stars(p: float) -> str:
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _paired_wilcoxon_fdr_by_R(
    records: list[dict],
    score_key: str = "f2",
    R_values: tuple[int, ...] = (2, 4, 8),
) -> dict[int, dict[str, float | int | bool | str]]:
    df = pd.DataFrame(records)
    if df.empty or not {"R", "method", "idx", score_key}.issubset(df.columns):
        return {}

    sub = df[df["R"].isin((1, *R_values)) & df["method"].isin(("cold", "global"))].copy()
    results: list[dict[str, float | int]] = []
    for R in R_values:
        tmp = sub[sub["R"] == R]
        try:
            wide = tmp.pivot(index="idx", columns="method", values=score_key).dropna()
        except ValueError:
            wide = tmp.pivot_table(index="idx", columns="method", values=score_key).dropna()
        if {"cold", "global"}.issubset(wide.columns) and len(wide) > 0:
            stat, p = wilcoxon(
                wide["global"],
                wide["cold"],
                zero_method="wilcox",
                alternative="greater",
            )
            results.append({"R": int(R), "stat": float(stat), "p_uncorrected": float(p), "n": int(len(wide))})

    if not results:
        return {}

    reject, pvals_corr, _, _ = multipletests(
        [row["p_uncorrected"] for row in results],
        method="fdr_bh",
    )
    out: dict[int, dict[str, float | int | bool | str]] = {}
    for row, p_corr, is_reject in zip(results, pvals_corr, reject):
        row["p_fdr"] = float(p_corr)
        row["reject_fdr"] = bool(is_reject)
        row["stars"] = _p_to_stars(float(p_corr))
        out[int(row["R"])] = row
    return out


def plot_real_rivers_with_simulation_points(
    all_records: list[dict],
    simulation_records_or_csv: list[dict] | str | Path,
    score_key: str = "f2",
    title: str | None = None,
    show_legend: bool = True,
    error_mode: str = "std",
    river_fill_alpha: float = 0.18,
    river_linewidth: float = 2.2,
    river_alpha: float = 0.95,
    river_point_size: float = 42,
    river_point_alpha: float = 0.95,
    river_point_color: str | None = "red",
    river_error_capsize: float = 3.0,
    real_draw_points: bool = True,
    real_point_size: float = 30,
    real_point_alpha: float = 0.28,
    real_point_spread: float = 0.035,
    sim_point_size: float = 34,
    sim_point_alpha: float = 0.9,
    sim_spread: float = 0.045,
    sim_draw_points: bool = True,
    sim_draw_summary: bool = True,
    sim_summary_show_markers: bool = True,
    sim_summary_linestyle: str = "-",
    sim_summary_linewidth: float = 1.8,
    sim_summary_alpha: float = 0.95,
    sim_summary_marker_size: float = 5.5,
    sim_summary_point_color: str | None = "dimgray",
    sim_error_capsize: float = 3.0,
    include_real_methods: tuple[str, ...] = ("cold", "global"),
    include_sim_methods: tuple[str, ...] = ("r1", "cold", "global"),
    alpha_by_date: dict[str, float] | None = None,
    show_real_significance: bool = True,
):
    """
    真实数据画 mean +/- error ribbon，real/simulation 原始点分开着色，
    simulation 叠加 mean +/- error 连线。
    """
    for rec in all_records:
        if "method" not in rec or "R" not in rec or score_key not in rec:
            raise ValueError(f"Real-data record missing required keys: {rec}")

    if isinstance(simulation_records_or_csv, (str, Path)):
        sim_records = load_compute_spiral_f2_details(
            simulation_records_or_csv, score_key=score_key
        )
    else:
        sim_records = list(simulation_records_or_csv)

    method_order = ("cold", "global")
    include_real_methods = tuple(str(method).strip().lower() for method in include_real_methods)
    include_sim_methods = tuple(str(method).strip().lower() for method in include_sim_methods)
    method_to_color = {"cold": "C0", "global": "C1"}
    sim_color_by_method = {"r1": "dimgray", "cold": "C0", "global": "C1"}
    sim_center_offset = {"r1": 0.0, "cold": -0.06, "global": 0.06}
    real_center_offset = {"cold": 0.0, "global": 0.0}
    alpha_by_date = alpha_by_date or {"1219": 0.25, "0210": 0.65, "0224": 0.8, "0306": 1.0}

    real_stats = _compute_mean_error_by_method_R(
        all_records,
        score_key=score_key,
        method_order=method_order,
        error_mode=error_mode,
        bridge_missing_R1_for_global=True,
        bridge_source_method="cold",
    )
    sim_stats = _compute_mean_error_by_method_R(
        sim_records,
        score_key=score_key,
        method_order=("r1", "cold", "global"),
        error_mode=error_mode,
        bridge_missing_R1_for_global=False,
    )

    real_groups: dict[tuple[str, int], list[dict]] = {}
    for rec in all_records:
        method = str(rec.get("method", "")).strip().lower()
        if method not in method_order:
            continue
        if "R" not in rec or score_key not in rec:
            continue

        value = float(rec[score_key])
        if not np.isfinite(value):
            continue

        R = int(rec["R"])
        real_groups.setdefault((method, R), []).append(rec)

    sim_groups: dict[tuple[str, int], list[dict]] = {}
    for rec in sim_records:
        method = str(rec.get("method", rec.get("recon", ""))).strip().lower()
        if method not in include_sim_methods:
            continue
        if "R" not in rec or score_key not in rec:
            continue

        value = float(rec[score_key])
        if not np.isfinite(value):
            continue

        R = int(rec["R"])
        sim_groups.setdefault((method, R), []).append(rec)

    all_R = sorted(
        {int(rec["R"]) for rec in all_records}
        | {R for _, R in sim_groups.keys()}
    )
    if not all_R:
        raise ValueError("No records to plot.")

    plt.figure(figsize=(7.8, 4.6))

    # 真实数据 river：先画带宽
    for method in method_order:
        if method not in include_real_methods:
            continue
        xs, mus, los, his = [], [], [], []
        for R in all_R:
            if R not in real_stats[method]:
                continue
            stat = real_stats[method][R]
            mu = float(stat["mean"])
            err = float(stat["error"])
            xs.append(R)
            mus.append(mu)
            los.append(max(0.0, mu - err))
            his.append(min(1.0, mu + err))

        if not xs:
            continue

        color = method_to_color[method]
        plt.fill_between(
            xs,
            los,
            his,
            color=color,
            alpha=river_fill_alpha,
            linewidth=0,
            zorder=1,
        )

    # real data 原始点：x 不做横向位移；颜色表示方法，marker 表示 visit。
    if real_draw_points:
        for method in method_order:
            if method not in include_real_methods:
                continue
            for R in all_R:
                group = real_groups.get((method, R), [])
                if not group:
                    continue

                center = float(R) + real_center_offset[method]
                for rec in group:
                    date_tag = str(rec.get("date_tag", ""))
                    visit_label = _visit_label_for_record(rec)
                    marker = VISIT_MARKER_BY_LABEL.get(visit_label, "o")
                    point_color = "dimgray" if int(rec["R"]) == 1 else method_to_color[method]
                    plt.scatter(
                        [center],
                        [float(rec[score_key])],
                        s=real_point_size,
                        alpha=alpha_by_date.get(date_tag, real_point_alpha),
                        color=point_color,
                        marker=marker,
                        edgecolors="white",
                        linewidths=0.4,
                        zorder=2,
                    )

    # 真实数据中心线与 summary 点 + error bar
    for method in method_order:
        if method not in include_real_methods:
            continue
        xs, mus, errs = [], [], []
        for R in all_R:
            if R not in real_stats[method]:
                continue
            stat = real_stats[method][R]
            xs.append(R)
            mus.append(float(stat["mean"]))
            errs.append(float(stat["error"]))

        if not xs:
            continue

        color = method_to_color[method]
        plt.errorbar(
            xs,
            mus,
            yerr=errs,
            color=color,
            linestyle="-",
            linewidth=river_linewidth,
            alpha=river_alpha,
            zorder=3,
            marker=None,
            elinewidth=max(1.0, river_linewidth - 0.2),
            capsize=river_error_capsize,
        )
        if river_point_size > 0:
            point_color = river_point_color if river_point_color is not None else color
            plt.scatter(
                xs,
                mus,
                s=river_point_size,
                color=point_color,
                alpha=river_point_alpha,
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

    sim_r1_stat = sim_stats.get("r1", {}).get(1)

    sim_mean_xs: list[float] = []
    sim_mean_ys: list[float] = []

    # simulation 统计线：cold/global 两条线都以 simulation 的 R1 均值作为起点
    for method in ("cold", "global"):
        if method not in include_sim_methods:
            continue

        xs, mus, errs, rs = [], [], [], []
        for R in all_R:
            if R == 1:
                stat = sim_r1_stat
            else:
                stat = sim_stats[method].get(R)

            if stat is None:
                continue

            xs.append(float(R))
            mus.append(float(stat["mean"]))
            errs.append(float(stat["error"]))
            rs.append(R)

        if sim_draw_summary and xs:
            plt.errorbar(
                xs,
                mus,
                yerr=errs,
                color=sim_color_by_method[method],
                linestyle=sim_summary_linestyle,
                linewidth=sim_summary_linewidth,
                alpha=sim_summary_alpha,
                marker=None,
                elinewidth=max(1.0, sim_summary_linewidth - 0.2),
                capsize=sim_error_capsize,
                zorder=5,
            )
            if sim_summary_show_markers:
                for x, y, R in zip(xs, mus, rs):
                    if R == 1:
                        continue
                    sim_mean_xs.append(x)
                    sim_mean_ys.append(y)

    if sim_draw_summary and sim_summary_show_markers:
        if sim_r1_stat is not None and "r1" in include_sim_methods:
            sim_mean_xs.append(1.0)
            sim_mean_ys.append(float(sim_r1_stat["mean"]))

        if sim_mean_xs:
            mean_point_color = (
                sim_summary_point_color if sim_summary_point_color is not None else "dimgray"
            )
            plt.scatter(
                sim_mean_xs,
                sim_mean_ys,
                s=sim_summary_marker_size**2,
                color=mean_point_color,
                edgecolors="white",
                linewidths=0.6,
                alpha=sim_summary_alpha,
                zorder=6,
            )

    # simulation 原始散点：按 (method, R) 分组后做对称抖动，避免完全重叠
    if sim_draw_points:
        for method in ("cold", "global"):
            if method not in include_sim_methods:
                continue
            for R in all_R:
                group = sim_groups.get((method, R), [])
                if not group:
                    continue

                ys = [float(rec[score_key]) for rec in group]
                center = float(R) + sim_center_offset[method]
                if len(group) == 1:
                    xs = np.asarray([center], dtype=float)
                else:
                    xs = np.asarray(
                        np.linspace(center - sim_spread, center + sim_spread, num=len(group)),
                        dtype=float,
                    )

                plt.scatter(
                    xs,
                    ys,
                    s=sim_point_size,
                    alpha=sim_point_alpha,
                    color=sim_color_by_method[method],
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=6,
                )

    plt.xticks(all_R, [f"R{r}" for r in all_R])
    plt.ylim(0, 1.05)
    plt.xlabel("R")
    plt.ylabel("F2 score (beta=2)" if score_key == "f2" else score_key)
    plt.grid(True, alpha=0.3)

    if title is None:
        title = f"Real-data rivers + simulation points ({score_key})"
    plt.title(title)

    if show_real_significance:
        sig_by_R = _paired_wilcoxon_fdr_by_R(all_records, score_key=score_key)
        real_vals = [
            float(rec[score_key])
            for rec in all_records
            if rec.get("method") in method_order and np.isfinite(float(rec[score_key]))
        ]
        y_sig = min(1.02, (max(real_vals) if real_vals else 1.0) + 0.025)
        for R in (2, 4, 8):
            if R in sig_by_R:
                plt.text(R, y_sig, str(sig_by_R[R]["stars"]), ha="center", va="bottom", fontsize=11)

    if show_legend:
        legend_handles = []

        for visit_label in ("V#1", "V#2", "V#4", "V#5"):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=VISIT_MARKER_BY_LABEL[visit_label],
                    markersize=7,
                    markerfacecolor="gray",
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    alpha=0.85,
                    label=visit_label,
                )
            )

        if "cold" in include_real_methods:
            legend_handles.append(
                Line2D([0], [0], color=method_to_color["cold"], linewidth=2.0, label="cold recon mean")
            )
        if "global" in include_real_methods:
            legend_handles.append(
                Line2D([0], [0], color=method_to_color["global"], linewidth=2.0, label="global recon mean")
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=7,
                markerfacecolor="dimgray",
                markeredgecolor="white",
                markeredgewidth=0.6,
                alpha=0.85,
                label="R1",
            )
        )

        if "cold" in include_sim_methods:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="o",
                    markersize=7,
                    markerfacecolor=method_to_color["cold"],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    alpha=sim_point_alpha,
                    label="simulation cold recon",
                )
            )
        if "global" in include_sim_methods:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="o",
                    markersize=7,
                    markerfacecolor=method_to_color["global"],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    alpha=sim_point_alpha,
                    label="simulation global recon",
                )
            )

        plt.legend(handles=legend_handles, frameon=True, ncol=2, fontsize=9)

    plt.tight_layout()


def plot_scores_across_bases(
    all_records: list[dict],
    score_key: str = "f2",        # "dice" or "f2"
    title: str | None = None,
    show_legend: bool = True,
    alpha_by_date: dict[str, float] | None = None,
    x_jitter_method: float = 0.00,
    x_jitter_date: float = 0.0,

    # mean 线：所有日期混一起，但 cold/global 各一条
    draw_mean_line: bool = True,
    mean_linestyle: str = "--",
    mean_linewidth: float = 1.2,
    mean_alpha: float = 0.95,
    mean_point_size: float = 45,   # scatter 的 s
    mean_point_color: str | None = None,  # None=跟 method 同色；"red"=强制红点
):
    # --- 容错 ---
    for r in all_records:
        r.setdefault("date_tag", "unknown")
        if "method" not in r or "R" not in r or score_key not in r:
            raise ValueError(f"Record missing required keys: {r}")

    all_R = sorted({int(r["R"]) for r in all_records})
    if not all_R:
        raise ValueError("No records to plot.")

    method_order = ["cold", "global"]
    method_to_color = {"cold": "C0", "global": "C1"}

    date_tags = sorted({r["date_tag"] for r in all_records})
    if alpha_by_date is None:
        if len(date_tags) == 1:
            alpha_by_date = {date_tags[0]: 0.85}
        else:
            lo, hi = 0.35, 0.90
            alphas = np.linspace(lo, hi, num=len(date_tags))
            alpha_by_date = {dt: float(a) for dt, a in zip(date_tags, alphas)}

    date_to_offset = {dt: (i - (len(date_tags)-1)/2.0) * x_jitter_date for i, dt in enumerate(date_tags)}
    method_to_offset = {"cold": -x_jitter_method, "global": +x_jitter_method}

    plt.figure(figsize=(7.6, 4.4))

    # --- 散点：method 颜色固定，日期用 alpha ---
    groups = {}
    for r in all_records:
        groups.setdefault((r["method"], r["date_tag"]), []).append(r)

    for method in method_order:
        for dt in date_tags:
            recs = groups.get((method, dt), [])
            if not recs:
                continue

            xs = [int(rr["R"]) + method_to_offset.get(method, 0.0) + date_to_offset.get(dt, 0.0) for rr in recs]
            ys = [float(rr[score_key]) for rr in recs]

            plt.scatter(
                xs, ys,
                marker="o",
                alpha=alpha_by_date.get(dt, 0.65),
                color=method_to_color.get(method, None),
                label=f"{method} ({dt})" if show_legend else None,
            )

    # --- 均值线：按 method 聚合（跨所有日期/idx） ---
    # --- 均值线：按 method 聚合（跨所有日期/idx）；global 缺 R1 时用 cold 的 R1 补点 ---
    if draw_mean_line:
        # 先把每个 method 的每个 R 的均值算出来，方便后面互相借点
        mean_by_method_R = {m: {} for m in method_order}  # mean_by_method_R[method][R] = mean
        for method in method_order:
            agg_R = {}
            for r in all_records:
                if r["method"] != method:
                    continue
                R = int(r["R"])
                agg_R.setdefault(R, []).append(float(r[score_key]))
            for R, vals in agg_R.items():
                if len(vals) > 0:
                    mean_by_method_R[method][R] = float(np.mean(vals))

        bridge_missing_R1_for_global = True
        bridge_source_method = "cold"  # 你也可以改成别的

        for method in method_order:
            xs, mus = [], []

            for R in all_R:
                if R in mean_by_method_R[method]:
                    mu = mean_by_method_R[method][R]
                else:
                    # 只对 global 缺 R1 的情况做桥接
                    if (
                        bridge_missing_R1_for_global
                        and method == "global"
                        and R == 1
                        and R in mean_by_method_R.get(bridge_source_method, {})
                    ):
                        mu = mean_by_method_R[bridge_source_method][R]
                    else:
                        continue

                xs.append(R + method_to_offset.get(method, 0.0))
                mus.append(mu)

            if len(xs) == 0:
                continue

            c_line = method_to_color.get(method, None)
            c_point = mean_point_color if mean_point_color is not None else c_line

            plt.plot(
                xs, mus,
                linestyle=mean_linestyle,
                linewidth=mean_linewidth,
                alpha=mean_alpha,
                color=c_line,
                zorder=5,
                label=f"{method} mean" if show_legend else None,
            )
            plt.scatter(xs, mus, s=mean_point_size, color=c_point, zorder=6)

    # --- 坐标轴与图例 ---
    plt.xticks(all_R, [f"R{r}" for r in all_R])
    plt.ylim(0, 1.05)
    plt.xlabel("R")
    plt.ylabel("F2 score (beta=2)" if score_key == "f2" else "Dice score")
    plt.grid(True, alpha=0.3)

    if title is None:
        title = f"{score_key.upper()} scatter vs R (all bases)"
    plt.title(title)

    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l and (l not in uniq):
                uniq[l] = h
        plt.legend(list(uniq.values()), list(uniq.keys()), frameon=True)

    plt.tight_layout()


def run_multi_base_pipeline(
    bases: list[str | Path],
    ref_img: nib.Nifti1Image,
    run_first_level_single_session,
    # 关键：每个 base 可能用不同的两个 R1 anchor；这里给两种用法
    anchors_by_date: dict[str, tuple[str, str]] | None = None,
    default_anchor: tuple[str, str] | None = None,
    run_cfg: dict[str, dict] = RUN_CFG_DEFAULT,
    beta: float = 2.0,
    thresh_kwargs: dict | None = None,
) -> tuple[list[BaseResult], list[dict]]:
    """
    多 BASE 处理入口。

    anchors_by_date:
      例如 {"0210": ("R1_66_cg", "R1_56_cg"), "1219": ("R1_66_cold", "R1_56_cold")}
      date_tag 来自 base.name 推断 (nifti_0210 -> 0210)

    default_anchor:
      如果某个 date_tag 不在 anchors_by_date，就用默认 anchor。

    返回:
      base_results: 每个 base 的完整结果（含 union/overlap）
      all_records:  用于绘图的扁平 records（已含 date_tag）
    """
    base_results: list[BaseResult] = []
    all_records: list[dict] = []

    bases = [Path(b) for b in bases]
    for base in bases:
        date_tag = infer_date_tag_from_base(base)

        if anchors_by_date and date_tag in anchors_by_date:
            r1a, r1b = anchors_by_date[date_tag]
        elif default_anchor is not None:
            r1a, r1b = default_anchor
        else:
            raise ValueError(
                f"No anchors provided for base {base} (date_tag={date_tag}). "
                f"Provide anchors_by_date or default_anchor."
            )

        folder_filter = make_real_invivo_folder_filter(date_tag, r1a, r1b)
        base_res, records = process_one_base(
            base=base,
            ref_img=ref_img,
            run_first_level_single_session=run_first_level_single_session,
            r1_folder_a=r1a,
            r1_folder_b=r1b,
            run_cfg=run_cfg,
            beta=beta,
            thresh_kwargs=thresh_kwargs,
            folder_filter=folder_filter,
        )

        # 给 records 加上 date_tag，便于总图分组
        for rec in records:
            rec["date_tag"] = base_res.date_tag

        base_results.append(base_res)
        all_records.extend(records)

    return base_results, all_records


def make_real_invivo_folder_filter(date_tag: str, r1_folder_a: str, r1_folder_b: str):
    """
    Real in-vivo folder policy:
    - 1219: keep the two R1 anchors plus canonical cold/global CS folders.
    - 0210/0224/0306: keep non-ORC folders only.
    """
    date_tag = str(date_tag)
    if date_tag == "1219":
        keep = {
            r1_folder_a,
            r1_folder_b,
            "R2_cs_cold",
            "R2_cs_global",
            "R4_cs_cold",
            "R4_cs_global",
            "R8_cs_cold",
            "R8_cs_global",
        }
        return lambda folder_name: folder_name in keep

    if date_tag in {"0210", "0224", "0306"}:
        return lambda folder_name: "orc" not in folder_name.lower()

    return lambda folder_name: True



def _style_full_box(ax, lw=1.0, grid=True):
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(lw)
    ax.tick_params(direction="in", top=True, right=True)
    if grid:
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(True, which="major", color="0.80", linewidth=0.6, alpha=0.6)
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            gl.set_linestyle((0, (2, 2)))
        ax.grid(True, which="minor", linestyle=(0, (1, 2)), color="0.90", linewidth=0.4, alpha=0.5)


def check_model_fit_at_peak(
    fmri_img,
    z_map,
    design_matrix,
    glm,
    reg_idx=0,
    title="Model fit",
    plot=True,
    pad_frac=0.08,
    y1_lim=None,
    y2_lim=None,
    sharex=True,
):
    """
    在 z_map 最大 z 值对应的 voxel（在 glm.masker_ 空间）检查 GLM 拟合：
    - 上图：Observed vs Predicted(all regressors)
    - 下图：指定 regressor 的贡献（X[:, reg_idx] * beta[reg_idx]）

    返回 dict：peak_idx, betas, predicted_full, predicted_reg, peak_z
    """
    masker = glm.masker_
    z_vals = masker.transform(z_map).ravel()
    if np.all(np.isnan(z_vals)):
        raise RuntimeError("z_map yields only NaNs through glm.masker_. Check masks / alignment.")
    peak_idx = int(np.nanargmax(z_vals))
    peak_z = float(z_vals[peak_idx])

    data_ts = masker.transform(fmri_img)         # (T, V)
    voxel_ts = data_ts[:, peak_idx]              # (T,)
    n_scans = voxel_ts.shape[0]

    X = design_matrix.values
    if X.shape[0] != n_scans:
        raise ValueError(f"Design matrix rows ({X.shape[0]}) != n_scans ({n_scans}).")

    betas = np.linalg.pinv(X) @ voxel_ts
    predicted_full = X @ betas
    predicted_reg = X[:, reg_idx] * betas[reg_idx]

    # regressor name
    reg_name = None
    try:
        reg_name = design_matrix.columns[reg_idx]
    except Exception:
        reg_name = f"reg_{reg_idx}"

    if plot:
        t = np.arange(n_scans)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6),
            sharex=sharex,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        )

        _style_full_box(ax1, lw=1.0)
        ax1.plot(t, voxel_ts, label="Observed", linewidth=1)
        ax1.plot(t, predicted_full, label="Predicted (all regs)", linewidth=1)
        ax1.set_ylabel("Signal (a.u.)")
        if y1_lim is not None:
            ax1.set_ylim(*y1_lim)
        else:
            y_left = np.concatenate([voxel_ts, predicted_full])
            y_left = y_left[np.isfinite(y_left)]
            if y_left.size > 0:
                y0, y1 = np.min(y_left), np.max(y_left)
                span = max(y1 - y0, 1e-12)
                ax1.set_ylim(y0 - pad_frac * span, y1 + pad_frac * span)
        ax1.legend(loc="best")
        ax1.set_title(f"{title} | peak z={peak_z:.2f}")

        _style_full_box(ax2, lw=1.0)
        ax2.plot(t, predicted_reg, label=f"Contribution: {reg_name}", linestyle="--", linewidth=1)
        ax2.set_xlabel("Time (frames)")
        ax2.set_ylabel("Contribution (a.u.)")
        if y2_lim is not None:
            ax2.set_ylim(*y2_lim)
        else:
            y_right = predicted_reg
            y_right = y_right[np.isfinite(y_right)]
            if y_right.size > 0:
                r0, r1 = np.min(y_right), np.max(y_right)
                rspan = max(r1 - r0, 1e-12)
                ax2.set_ylim(r0 - pad_frac * rspan, r1 + pad_frac * rspan)
        ax2.legend(loc="best")
        plt.show()

    return {
        "peak_idx": peak_idx,
        "peak_z": peak_z,
        "betas": betas,
        "predicted_full": predicted_full,
        "predicted_reg": predicted_reg,
    }


def plot_motion_params(
    motion: np.ndarray,
    tr: float | None = None,
    title: str = "Head Motion Parameters (SPM Realignment)",
    ylim_trans: tuple[float, float] | None = (-3, 3),
    ylim_rot: tuple[float, float] | None = (-0.5, 0.5),
):
    """
    motion: shape (T,6). 前3列 translation(mm)，后3列 rotation(deg)。
    tr: 若提供，用秒作为横轴；否则用 frame index。
    """
    motion = np.asarray(motion)
    if motion.ndim != 2 or motion.shape[1] < 6:
        raise ValueError(f"motion must be (T,6+). got {motion.shape}")

    trans = motion[:, :3]
    rot = motion[:, 3:6]
    n_frames = motion.shape[0]

    if tr is None:
        x = np.arange(n_frames)
        x_label = "Frame"
    else:
        x = np.arange(n_frames) * float(tr)
        x_label = "Time (s)"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
    )

    ax1.plot(x, trans[:, 0], linewidth=1.2, label="x")
    ax1.plot(x, trans[:, 1], linewidth=1.2, label="y")
    ax1.plot(x, trans[:, 2], linewidth=1.2, label="z")
    ax1.set_title(title)
    ax1.set_ylabel("Translation (mm)")
    ax1.legend(ncol=3, frameon=False, loc="upper left")
    if ylim_trans is not None:
        ax1.set_ylim(*ylim_trans)
    ax1.grid(True, alpha=0.25, linestyle="--")

    ax2.plot(x, rot[:, 0], linewidth=1.2, label="pitch")
    ax2.plot(x, rot[:, 1], linewidth=1.2, label="roll")
    ax2.plot(x, rot[:, 2], linewidth=1.2, label="yaw")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Rotation (deg)")
    ax2.legend(ncol=3, frameon=False, loc="upper left")
    if ylim_rot is not None:
        ax2.set_ylim(*ylim_rot)
    ax2.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()
    plt.show()
    return fig

def p2z(p_value: float, double_side: bool = False) -> float:
    if double_side:
        p_value = p_value / 2
    # Calculate the z-score for the one-sided p-value
    z_score = norm.ppf(1 - p_value)
    return z_score
p2z(0.001)
p2z(0.01)

def _init_roc_plot(xlabel, ylabel, ax, luck=True):
    eps = 0.05
    ax.set_xlim(-eps, 1 + eps)
    ax.set_ylim(-eps, 1 + eps)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid("on")
    if luck:
        ax.plot([[0, 0], [1, 1]], c="gray", ls="dashed", lw=1)
    return ax

def _add_pvalue_marker(ax, x, y, thresh,  p_values, **kwargs):
    line = ax.plot(x,y, **kwargs)
    for m, p in p_values.items():
        z = p2z(p, double_side=False)
        idx = np.argmin(abs(thresh - z))
        ax.plot(x[idx], y[idx], marker=m, mew=2, c=line[0].get_color())
    return line

def roc_plot(results, ax, label, **kwargs):
    _init_roc_plot("FPR", "TPR", ax=ax, luck=True)

    threshs = np.array(results["tresh"])
    fpr, tpr = results["fpr"], results["tpr"]
    line = _add_pvalue_marker(ax, recall, fpr, tpr, {"o":1e-3, "x":1e-2})
    auc_val = auc(fpr, tpr)
    bacc_val = bacc(tpr[p001], fpr[p001])

    return dict(auc=f"{auc_val:.4}", bacc=f"{bacc_val:.3}", label=label, line=line)


def pr_plot(results, ax, label, **kwargs):
    _init_roc_plot("Recall", "Precision", ax=ax, luck=False)

    threshs = np.array(results["tresh"][1:])
    idx = np.argmin(abs(threshs - p2z(0.001))) # the closest threshold idx to the pvalue
    p01 = np.argmin(abs(threshs - p2z(0.01))) #no need for min for one side?
    p001 = np.min(np.where(threshs == threshs[idx])) #no need for min for one side?
    tpr = recall = results["tpr"][1:]
    fpr = results["fpr"][1:]
    tp = results["tp"][1:]
    fp = results["fp"][1:]
    precision = np.array(tp) / (np.array(tp) + np.array(fp))
    line = _add_pvalue_marker(ax, recall, precision, threshs[1:-1], {"o":1e-3,"x":1e-2}, **kwargs)
    # line = ax.plot(recall, precision, label=label, **kwargs)
    # ax.plot(recall[p001], precision[p001], marker="o", c=line[0].get_color())
    auc_val = auc(recall, precision)
    bacc_val = bacc(tpr[idx], fpr[idx], adjusted=True)
    return dict(auc=f"{auc_val:.4f}", bacc=f"{bacc_val:.4f}",label=label, line=line)
def pr_plot_df(row, ax, label, **kwargs):
    _init_roc_plot("Recall", "Precision", ax=ax, luck=False)

    threshs = np.array(row["results.tresh"])[1:-1]
    idx = np.argmin(abs(threshs[:-1] - p2z(0.01)))

    p001 = np.min(np.where(threshs == threshs[idx]))
    tpr = recall = row["results.tpr"][1:]
    fpr = row["results.fpr"][1:-1]
    tp = row["results.tp"][1:-1]
    fp = row["results.fp"][1:-1]
    precision = np.array(tp) / (np.array(tp) + np.array(fp))

    line = _add_pvalue_marker(ax, recall, precision, threshs, {"o":1e-3, "x": 1e-2}, **kwargs)
    # line = ax.plot(recall, precision, label=label, **kwargs)
    # ax.plot(recall[p001], precision[p001], marker="o", c=line[0].get_color())
    auc_val = auc(recall, precision)
    bacc_val = bacc(tpr[idx], fpr[idx], adjusted=True)
    return dict(auc=f"{auc_val:.4f}",  label=label, line=line)

def get_legend(infos):
    fontsize = 10
    handleheight = 1
    handlelength = 2
    legend_handler_map = Legend._default_handler_map
    # The approximate height and descent of text. These values are
    # only used for plotting the legend handle.
    descent = 0.35 * fontsize * (handleheight - 0.5)  # heuristic.
    height = fontsize * handleheight
    width = handlelength * fontsize
    handles_boxes = [
        DrawingArea(width=width, height=height, xdescent=0, ydescent=descent)
        for _ in range(len(infos))
    ]

    for h, i in zip(handles_boxes, infos):
        xdata = [0, 0.5]
        l = Line2D([0, width / 2, width], [descent + 1] * 3, markevery=[1])
        l.update_from(i["line"][0])
        l.set_transform(h.get_transform())
        h.add_artist(l)

    packer = HPacker(
        children=[
            VPacker(
                children=[
                    HPacker(children=[TextArea(l["label"]), h], sep=5)
                    for h, l in zip(handles_boxes, infos)
                ],
                sep=1,
                align="right",
            ),
             #VPacker(children=[TextArea("Nc")] + [TextArea(f'{l["Ncoils"]}') for l in infos], sep=1, align="right"),
            #  VPacker(children=[TextArea("SNR")] + [TextArea(f'{l["SNR"]}') for l in infos], sep=1, align="right"),
            VPacker(
                children=[VPacker(children=[TextArea(infos[0]["auc"])], height=18)]
                + [TextArea(l["auc"]) for l in infos[1:]],
                sep=1,
                align="right",
            ),
            VPacker(
                children=[VPacker(children=[TextArea(infos[0]["bacc"])], height=18)]
                + [TextArea(l["bacc"]) for l in infos[1:]],
                sep=1,
                align="right",
            ),
        ],
        align="bottom",
        sep=5,
    )

    legend = AnchoredOffsetbox(
        child=VPacker(children=[packer], align="right", sep=5), loc="lower left"
    )
    legend.patch.set_alpha(0.7)
    return legend
