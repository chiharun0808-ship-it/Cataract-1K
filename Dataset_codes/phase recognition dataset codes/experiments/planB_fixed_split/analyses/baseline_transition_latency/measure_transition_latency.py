"""
既存モデル（スライディングウィンドウ版 0914_10_46）の予測ログから、
「フェーズが切り替わってから、予測が新しいフェーズに追いつくまでの時間」を測る。

案B の入力条件（サンプリング間隔・時間幅）を決めるための判断材料。推論も学習もせず、
既存の予測ログと注釈ファイルを読むだけ。既存のファイルは変更しない。

## 定義（2026-10-04 にユーザーと合意。結果を見てから変えない）
- 対象: test 11症例の予測ログ（1秒ごと、平滑化なし）
  attention_analysis/streaming_attention/pred_logs/<case>.pred_log.csv
  （result/epochs_35/sliding_window_checkpoints_0914_10_46 の出力）
- 正解の時系列: 注釈の区間を 0.01秒刻みで並べ、utils/gt_utils.true_phase_at と同じ規則
  （区間が重なるときは先に書かれた区間を優先、どの区間にも入らない時刻は idle）でラベルを付ける。
- 境目: 正解ラベルが変わる時刻 b。主な分析は idle との切り替わりも含む全ての境目で、
  「idle に入る境目」と「手術フェーズに入る境目」にも分けて集計する。
  補助として「手術フェーズどうしの切り替わり」も集計する。注釈では全ての手術フェーズの間に
  idle が挟まっているため（2026-10-04 に確認）、これは「idle を挟んで、直前の手術フェーズと
  違うフェーズが始まる時点」と定義する（当初の書き方「idle を含まない境目」では0件になるため、
  意図に合わせて定義を明確にした。閾値や区切りは変えていない）。
- 新しい区間: 境目 b から次の境目（または動画の終わり）e まで。
- 追いつくまでの時間（K = 1, 2, 3, 5 秒の4通り。K=1 が「初めて当たるまで」）:
  区間 [b, e) の中の記録時刻 t で、t から K 点続けて予測 = 新しいフェーズ となる最初の t を探し、
  t - b を遅れとする。K 点続けた記録が全て区間の中にあることを条件とする。
  結果は3つに分ける:
    caught         : 追いついた
    not_caught     : 区間の中に K 点以上の記録があるのに、K 点続けて当たらなかった
    not_evaluable  : 区間の中の記録が K 点未満で、そもそも判定できない（区間が短い）
- 境目からの経過時間ごとの一致率: 各記録時刻について、直前の境目からの経過時間で区切って
  一致率を出す（0-1, 1-2, ..., 11-12, 12秒以上）。最初の境目より前の記録（動画の冒頭）は除く。
  参考線: 窓（30枚、1/3秒間隔）のうち新しいフェーズの映像が占める割合（計算上の値）。
- 区間の長さの区切り: 3秒未満 / 3-10秒 / 10秒以上
- フェーズごとの区間の長さの中央値は train 34症例の注釈から計算する（test の正解は使わない）。
- 全体の値の95%信頼区間: 症例を選び直すブートストラップ（2000回、乱数の固定値 0）。

## 実行方法（phase recognition dataset codes/ 直下で）
    python experiments/planB_fixed_split/analyses/baseline_transition_latency/measure_transition_latency.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLAN_DIR = HERE.parents[1]                      # experiments/planB_fixed_split
CODES_DIR = PLAN_DIR.parents[1]                 # phase recognition dataset codes
DATASET_DIR = CODES_DIR.parents[1] / "Phase_recognition_dataset"
sys.path.insert(0, str(CODES_DIR))

from utils.gt_utils import load_ground_truth_intervals  # noqa: E402  (読むだけ)

PRED_LOG_DIR = CODES_DIR / "attention_analysis" / "streaming_attention" / "pred_logs"
SPLIT_CSV = PLAN_DIR / "data" / "case_split_56cases.csv"
GT_CSV = DATASET_DIR / "phase_recognition_annotations" / "{case}" / "{case}_annotations_phases.csv"
VIDEO = DATASET_DIR / "dataset_videos" / "phase_recognition" / "{case}.mp4"
OUT_CSV = HERE / "csv"
OUT_PLOT = HERE / "plots"

PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle",
]
K_VALUES = [1, 2, 3, 5]
FINE_STEP = 0.01
ELAPSED_EDGES = list(range(0, 13)) + [np.inf]          # 0-1, ..., 11-12, 12+
LENGTH_BINS = [(0, 3, "<3s"), (3, 10, "3-10s"), (10, np.inf, ">=10s")]
FRAME_SPACING = 1 / 3
SEQ_LEN = 30
N_BOOT, SEED = 2000, 0


def video_duration(case):
    cap = cv2.VideoCapture(str(VIDEO).format(case=case))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps


def fine_timeline(case):
    """0.01秒刻みの正解ラベル。true_phase_at と同じく、先に書かれた区間を優先する。"""
    intervals = load_ground_truth_intervals(str(GT_CSV).format(case=case), PHASES)
    t = np.arange(0, video_duration(case), FINE_STEP)
    labels = np.full(len(t), "idle", dtype=object)
    assigned = np.zeros(len(t), dtype=bool)
    for start, end, name in intervals:
        hit = (~assigned) & (t >= start) & (t <= end)
        labels[hit] = name
        assigned |= hit
    return t, labels


def segments(t, labels):
    """[(start, end, label), ...]。end は次の区間の開始（最後は動画の終わり）。"""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(t)]])
    end_times = [t[e] if e < len(t) else t[-1] + FINE_STEP for e in ends]
    return [(t[s], et, labels[s]) for s, et in zip(starts, end_times)]


def latency(log, b, e, new_phase, k):
    inside = log[(log["time_sec"] >= b) & (log["time_sec"] < e)]
    if len(inside) < k:
        return "not_evaluable", np.nan
    hits = (inside["pred_phase"] == new_phase).to_numpy()
    times = inside["time_sec"].to_numpy()
    for i in range(len(hits) - k + 1):
        if hits[i:i + k].all():
            return "caught", times[i] - b
    return "not_caught", np.nan


def boot_ci(df, case_col, stat, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    cases = df[case_col].unique()
    groups = {c: g for c, g in df.groupby(case_col)}
    vals = []
    for _ in range(n):
        pick = rng.choice(cases, size=len(cases), replace=True)
        v = stat(pd.concat([groups[c] for c in pick]))
        if not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def pct(x):
    return 100 * x


def main():
    split = pd.read_csv(SPLIT_CSV)
    test_cases = sorted(split.loc[split["split"] == "test", "case_id"])
    train_cases = sorted(split.loc[split["split"] == "train", "case_id"])

    # --- train 34症例: フェーズごとの区間の長さの中央値 ---
    seg_rows = []
    for case in train_cases:
        t, lab = fine_timeline(case)
        for s, e, name in segments(t, lab):
            seg_rows.append({"case_id": case, "phase": name, "duration_sec": e - s})
    train_seg = pd.DataFrame(seg_rows)
    train_median = train_seg.groupby("phase")["duration_sec"].median()

    # --- test 11症例: 境目ごとの遅れ ---
    rows, step_rows, log_hashes = [], [], {}
    for case in test_cases:
        log_path = PRED_LOG_DIR / f"{case}.pred_log.csv"
        log_hashes[log_path.name] = hashlib.sha256(log_path.read_bytes()).hexdigest()
        log = pd.read_csv(log_path)
        t, lab = fine_timeline(case)
        segs = segments(t, lab)
        last_surgical = None
        for prev, cur in zip(segs[:-1], segs[1:]):
            b, e, new_phase = cur
            if prev[2] != "idle":
                last_surgical = prev[2]          # cur の直前にあった手術フェーズ
            surgical_change = (new_phase != "idle" and last_surgical is not None
                               and last_surgical != new_phase)
            row = {"case_id": case, "boundary_sec": round(b, 2), "prev_phase": prev[2],
                   "new_phase": new_phase, "new_segment_sec": round(e - b, 2),
                   "into_idle": new_phase == "idle",
                   "surgical_phase_change": surgical_change,
                   "prev_surgical_phase": last_surgical if new_phase != "idle" else None}
            for k in K_VALUES:
                status, lat = latency(log, b, e, new_phase, k)
                row[f"K{k}_status"], row[f"K{k}_latency_sec"] = status, lat
            rows.append(row)

        # 経過時間ごとの一致率（最初の境目より前は除く）
        bounds = np.array([s for s, _, _ in segs[1:]])
        for _, r in log.iterrows():
            prior = bounds[bounds <= r["time_sec"]]
            if len(prior) == 0:
                continue
            step_rows.append({"case_id": case, "elapsed_sec": r["time_sec"] - prior[-1],
                              "correct": r["pred_phase"] == r["true_phase"]})

    bd = pd.DataFrame(rows)
    steps = pd.DataFrame(step_rows)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_PLOT.mkdir(parents=True, exist_ok=True)
    bd.to_csv(OUT_CSV / "boundaries.csv", index=False)

    def summarize(df, label):
        out = []
        for k in K_VALUES:
            st = df[f"K{k}_status"]
            ev = df[st != "not_evaluable"]
            lat = df[f"K{k}_latency_sec"]
            within = {w: (lambda d, w=w: ((d[f"K{k}_status"] == "caught") & (d[f"K{k}_latency_sec"] <= w)).sum()
                          / max((d[f"K{k}_status"] != "not_evaluable").sum(), 1)) for w in (2, 5, 10)}
            caught_rate = lambda d: (d[f"K{k}_status"] == "caught").sum() / max((d[f"K{k}_status"] != "not_evaluable").sum(), 1)
            med = lambda d: d.loc[d[f"K{k}_status"] == "caught", f"K{k}_latency_sec"].median()
            r = {"group": label, "K_sec": k, "n_boundaries": len(df),
                 "n_not_evaluable": int((st == "not_evaluable").sum()),
                 "n_evaluable": len(ev),
                 "caught_pct_of_evaluable": pct(caught_rate(df)),
                 "not_caught_pct_of_evaluable": pct(1 - caught_rate(df)) if len(ev) else np.nan,
                 "median_latency_sec_caught": med(df)}
            if df["case_id"].nunique() > 1:
                lo, hi = boot_ci(df, "case_id", med)
                r["median_latency_ci95"] = f"[{lo:.1f}, {hi:.1f}]"
                lo, hi = boot_ci(df, "case_id", caught_rate)
                r["caught_pct_ci95"] = f"[{pct(lo):.1f}, {pct(hi):.1f}]"
            for w, fn in within.items():
                r[f"caught_within_{w}s_pct_of_evaluable"] = pct(fn(df))
            out.append(r)
        return out

    summ = (summarize(bd, "all_boundaries")
            + summarize(bd[bd["into_idle"]], "into_idle")
            + summarize(bd[~bd["into_idle"]], "into_surgical_phase")
            + summarize(bd[bd["surgical_phase_change"]], "surgical_phase_change_only"))
    pd.DataFrame(summ).round(2).to_csv(OUT_CSV / "summary_by_K.csv", index=False)

    # 区間の長さ別
    len_rows = []
    for lo_, hi_, name in LENGTH_BINS:
        sub = bd[(bd["new_segment_sec"] >= lo_) & (bd["new_segment_sec"] < hi_)]
        len_rows += summarize(sub, f"new_segment_{name}")
    pd.DataFrame(len_rows).round(2).to_csv(OUT_CSV / "by_new_segment_length.csv", index=False)

    # フェーズ別（train の区間の長さの中央値を横に並べる）
    ph_rows = []
    for ph in PHASES:
        sub = bd[bd["new_phase"] == ph]
        if sub.empty:
            continue
        r = {"new_phase": ph, "train_median_segment_sec": round(train_median.get(ph, np.nan), 2),
             "n_boundaries_test": len(sub)}
        for k in K_VALUES:
            st = sub[f"K{k}_status"]
            ev = (st != "not_evaluable").sum()
            r[f"K{k}_not_evaluable"] = int((st == "not_evaluable").sum())
            r[f"K{k}_caught_pct_of_evaluable"] = round(pct((st == "caught").sum() / ev), 1) if ev else np.nan
            r[f"K{k}_median_latency_sec"] = sub.loc[st == "caught", f"K{k}_latency_sec"].median()
        ph_rows.append(r)
    pd.DataFrame(ph_rows).round(2).to_csv(OUT_CSV / "by_phase.csv", index=False)

    # 症例別
    case_rows = []
    for case, sub in bd.groupby("case_id"):
        log = pd.read_csv(PRED_LOG_DIR / f"{case}.pred_log.csv")
        r = {"case_id": case, "n_boundaries": len(sub),
             "overall_accuracy_pct": round(pct((log["pred_phase"] == log["true_phase"]).mean()), 1)}
        for k in K_VALUES:
            st = sub[f"K{k}_status"]
            ev = (st != "not_evaluable").sum()
            r[f"K{k}_caught_pct_of_evaluable"] = round(pct((st == "caught").sum() / ev), 1) if ev else np.nan
            r[f"K{k}_median_latency_sec"] = sub.loc[st == "caught", f"K{k}_latency_sec"].median()
        case_rows.append(r)
    pd.DataFrame(case_rows).round(2).to_csv(OUT_CSV / "by_case.csv", index=False)

    # 経過時間ごとの一致率
    labels = [f"{a}-{b}s" if np.isfinite(b) else f">={a}s" for a, b in zip(ELAPSED_EDGES[:-1], ELAPSED_EDGES[1:])]
    steps["bin"] = pd.cut(steps["elapsed_sec"], bins=ELAPSED_EDGES, right=False, labels=labels)
    el_rows = []
    for i, lab_ in enumerate(labels):
        sub = steps[steps["bin"] == lab_]
        acc = sub["correct"].mean()
        lo, hi = boot_ci(sub, "case_id", lambda d: d["correct"].mean())
        start = ELAPSED_EDGES[i]
        ref = min(int((start + 0.5) // FRAME_SPACING) + 1, SEQ_LEN) / SEQ_LEN
        el_rows.append({"elapsed_bin": lab_, "n_steps": len(sub), "accuracy_pct": pct(acc),
                        "ci95_low": pct(lo), "ci95_high": pct(hi),
                        "reference_new_phase_share_of_window_pct": pct(ref)})
    el = pd.DataFrame(el_rows).round(1)
    el.to_csv(OUT_CSV / "accuracy_by_elapsed.csv", index=False)

    # 図: 経過時間ごとの一致率と参考線（1つの軸、0-100%）
    x = np.arange(len(el)) + 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.plot(x, el["reference_new_phase_share_of_window_pct"], color="#8a8a85", lw=1.5, ls="--",
            label="Reference: share of the 9.67-s window already in the new phase")
    ax.errorbar(x, el["accuracy_pct"], yerr=[el["accuracy_pct"] - el["ci95_low"], el["ci95_high"] - el["accuracy_pct"]],
                color="#2a78d6", lw=2, marker="o", ms=5, capsize=3, label="Observed per-second accuracy (95% CI, case bootstrap)")
    ax.set_xticks(x); ax.set_xticklabels(el["elapsed_bin"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Time since the ground-truth phase change")
    ax.set_title("Accuracy recovers within ~1 s of a phase change, long before the window turns over\n"
                 "(test 11 cases, model 0914_10_46)", fontsize=10)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PLOT / "accuracy_by_elapsed.png", facecolor=fig.get_facecolor())

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(CODES_DIR)).replace("\\", "/"),
        "model_of_pred_logs": "result/epochs_35/sliding_window_checkpoints_0914_10_46/best_model.pth",
        "pred_log_sha256": log_hashes,
        "split_file": str(SPLIT_CSV.relative_to(CODES_DIR)).replace("\\", "/"),
        "test_cases": test_cases, "n_train_cases_for_durations": len(train_cases),
        "K_values_sec": K_VALUES, "bootstrap": {"n": N_BOOT, "seed": SEED},
        "n_boundaries": len(bd), "n_steps_after_first_boundary": len(steps),
    }
    (HERE / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"境目の数: {len(bd)}（test 11症例）")
    print(pd.DataFrame(summ).round(2).to_string(index=False))
    print(el.to_string(index=False))


if __name__ == "__main__":
    main()
