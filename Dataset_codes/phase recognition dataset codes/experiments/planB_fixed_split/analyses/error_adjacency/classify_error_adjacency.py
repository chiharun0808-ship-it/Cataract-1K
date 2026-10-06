"""
現行モデル（91.4%）と再調査のモデル（82.8%）の誤りを、隣接・非隣接に分けて数える（読むだけの分析。2026-10-05）。

定義1（既存の attention 分析と同じ。streaming_attention_eval.classify_confusion）:
  - idle が絡む誤り（正解か予測が idle）は別枠「idle絡み」
  - 手術フェーズどうし: 標準の手術順序で位置の差が1なら「隣接」、2以上なら「非隣接」
定義2（その動画の実際の流れに基づく。時間的な隣接）:
  - 動画の正解を「同じラベルが続く区間」に分け、誤りの時点が入っている区間の直前・直後の区間を調べる
  - 予測が、直前・直後の区間のラベル（idle を含む）、または idle を飛ばした直前・直後の手術フェーズと
    同じなら「時間的に隣接」、どれでもなければ「時間的に非隣接」
    （ほとんどのフェーズの間には idle 区間がはさまるので、idle を飛ばした前後も隣接として扱う）
  - あわせて、誤りの時点から最も近い区間の境目までの秒数を記録する
出力: csv/errors.csv（全誤り）、csv/summary.csv（モデル × 定義 × 区分の件数）、summary.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from lib.paths import BASELINE_PRED_LOG_DIR, RUNS_DIR, TEST_CASES  # noqa: E402

RUN = RUNS_DIR / "2026-10-04_defectfree_f30_i333ms_s5s_vitprep_timesformer_seed0"
ORDER = ["Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection", "Phacoemulsification",
         "Irrigation_Aspiration", "CapsulePulishing", "LensImplantation", "LensPositioning",
         "Viscoelastic_Suction", "Anterior_ChamberFlushing", "Tonifying_Antibiotics"]
MODELS = {"baseline_91.4": BASELINE_PRED_LOG_DIR, "defectfree": RUN / "streaming_eval" / "pred_logs"}


def def1(true, pred):
    if true == "idle" or pred == "idle":
        return "idle絡み"
    return "隣接" if abs(ORDER.index(true) - ORDER.index(pred)) <= 1 else "非隣接"


def segments(true_series, times):
    segs, start = [], 0
    for i in range(1, len(true_series) + 1):
        if i == len(true_series) or true_series[i] != true_series[start]:
            segs.append((start, i - 1, true_series[start], times[start], times[i - 1]))
            start = i
    return segs


def main():
    rows = []
    for model, log_dir in MODELS.items():
        for c in TEST_CASES:
            d = pd.read_csv(Path(log_dir) / f"{c}.pred_log.csv")
            tr, pr, t = d.true_phase.tolist(), d.pred_phase.tolist(), d.time_sec.tolist()
            segs = segments(tr, t)
            seg_of = {}
            for k, (a, b, *_rest) in enumerate(segs):
                for i in range(a, b + 1):
                    seg_of[i] = k
            for i in range(len(tr)):
                if tr[i] == pr[i]:
                    continue
                k = seg_of[i]
                prev_lab = segs[k - 1][2] if k > 0 else None
                next_lab = segs[k + 1][2] if k + 1 < len(segs) else None
                # idle を飛ばした直前・直後の手術フェーズ（ほとんどのフェーズの間には idle がはさまるため）
                prev_act = next((s[2] for s in reversed(segs[:k]) if s[2] not in ("idle", tr[i])), None)
                next_act = next((s[2] for s in segs[k + 1:] if s[2] not in ("idle", tr[i])), None)
                neighbors = {prev_lab, next_lab, prev_act, next_act}
                temporal = "時間的に隣接" if pr[i] in neighbors else "時間的に非隣接"
                start_t, end_t = segs[k][3], segs[k][4]
                dist = min(t[i] - start_t, end_t - t[i] + 1.0)
                kind = ("正解idle→予測別フェーズ" if tr[i] == "idle" else
                        "正解別フェーズ→予測idle" if pr[i] == "idle" else "手術フェーズどうし")
                rows.append({"model": model, "case_id": c, "time_sec": t[i], "true": tr[i], "pred": pr[i],
                             "kind": kind, "def1": def1(tr[i], pr[i]), "def2": temporal,
                             "prev_segment": prev_lab, "next_segment": next_lab,
                             "sec_to_nearest_boundary": round(dist, 2)})
    df = pd.DataFrame(rows)
    (HERE / "csv").mkdir(exist_ok=True)
    df.to_csv(HERE / "csv" / "errors.csv", index=False, encoding="utf-8")

    s1 = df.groupby(["model", "def1"]).size().unstack(fill_value=0)
    s2 = df.groupby(["model", "kind", "def2"]).size().unstack(fill_value=0)
    near = df.assign(near=df.sec_to_nearest_boundary <= 10).groupby(["model", "kind"])["near"].mean().round(3)
    s1.to_csv(HERE / "csv" / "summary_def1.csv")
    s2.to_csv(HERE / "csv" / "summary_def2_by_kind.csv")
    nonadj = df[df.def2 == "時間的に非隣接"].groupby(["model", "true", "pred"]).size().sort_values(ascending=False)
    nonadj.to_csv(HERE / "csv" / "def2_nonadjacent_pairs.csv")
    out = {"total_errors": df.groupby("model").size().to_dict(),
           "def1": s1.to_dict(orient="index"),
           "share_within_10s_of_boundary_by_kind": {f"{m}|{k}": v for (m, k), v in near.items()}}
    json.dump(out, open(HERE / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("定義1（既存）\n", s1.to_string(), "\n")
    print("定義2（その動画の流れ）\n", s2.to_string(), "\n")
    print("境目から10秒以内の割合\n", near.to_string(), "\n")
    for m in MODELS:
        print(f"時間的に非隣接の組み合わせ上位（{m}）")
        print(nonadj.loc[m].head(8).to_string(), "\n")


if __name__ == "__main__":
    main()
