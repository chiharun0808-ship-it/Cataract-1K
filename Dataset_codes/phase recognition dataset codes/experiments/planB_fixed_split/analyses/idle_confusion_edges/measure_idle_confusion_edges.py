"""
候補C: 再調査のモデルで増えた idle の取り違えが、「中央の切り出しで画面の端が切れたこと」と
関係していそうかを確かめる（読むだけの分析。学習・推論はしない。2026-10-05 ユーザー承認）。

- 対象（新たな取り違え）: test 11症例で、現行モデルは正解・再調査のモデルは不正解、かつ idle が絡む時点
  （正解 idle → 予測が別フェーズ、または 正解が別フェーズ → 予測 idle）
- 対照: 両モデルとも正解だった時点から、同じ症例で同じ数をランダムに選ぶ（乱数の固定値 0）
- 各時点の窓の末尾フレームを元動画から読み、切り出しで切れる部分（x<88, x>=424, y<24, y>=360）について
  「明るい画素（グレー値>=100）のうち切れる部分に入る割合」を計算して、対象と対照で比べる
- 画像を枠付きで並べ、目で確かめられるようにする
注意: 見るのは窓の最後の1枚だけ。窓の30枚のうち、ほかのフレームの端に器具があったかは見ていない。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from lib.paths import BASELINE_PRED_LOG_DIR, RUNS_DIR, TEST_CASES, VIDEO_TEMPLATE  # noqa: E402

RUN = RUNS_DIR / "2026-10-04_defectfree_f30_i333ms_s5s_vitprep_timesformer_seed0"
X0, Y0, X1, Y1 = 88, 24, 424, 360


def bright_lost(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = g >= 100
    inside = np.zeros_like(b)
    inside[Y0:Y1, X0:X1] = True
    return float((b & ~inside).sum() / b.sum()) if b.sum() else 0.0


def main():
    rng = np.random.default_rng(0)
    targets = []
    for c in TEST_CASES:
        a = pd.read_csv(BASELINE_PRED_LOG_DIR / f"{c}.pred_log.csv")
        b = pd.read_csv(RUN / "streaming_eval" / "pred_logs" / f"{c}.pred_log.csv")
        d = pd.DataFrame({"t": b.time_sec, "true": b.true_phase, "base": a.pred_phase, "new": b.pred_phase})
        new_err = (d.base == d.true) & (d.new != d.true) & ((d.true == "idle") | (d.new == "idle"))
        both_ok = (d.base == d.true) & (d.new == d.true)
        err = d[new_err].assign(group="new_idle_confusion")
        ctrl_pool = d[both_ok]
        ctrl = ctrl_pool.iloc[rng.choice(len(ctrl_pool), size=min(len(err), len(ctrl_pool)), replace=False)].assign(group="control_both_correct")
        for _, r in pd.concat([err, ctrl]).iterrows():
            targets.append({"case_id": c, **r.to_dict()})
    df = pd.DataFrame(targets)
    df["kind"] = np.where(df.group == "control_both_correct", "control",
                          np.where(df["true"] == "idle", "idle_to_active", "active_to_idle"))

    vals, crops = [], {}
    for c, g in df.groupby("case_id"):
        cap = cv2.VideoCapture(VIDEO_TEMPLATE.format(case=c))
        fps = cap.get(cv2.CAP_PROP_FPS)
        want = {int(round(t * fps)): i for i, t in zip(g.index, g.t)}
        f, last = 0, max(want)
        while f <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if f in want:
                i = want[f]
                vals.append((i, bright_lost(frame)))
                crops[i] = frame
            f += 1
        cap.release()
    for i, v in vals:
        df.loc[i, "bright_lost"] = v
    (HERE / "csv").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)
    df.to_csv(HERE / "csv" / "frames.csv", index=False)

    summ = df.groupby("kind")["bright_lost"].describe(percentiles=[0.5, 0.9]).round(4)
    summ.to_csv(HERE / "csv" / "summary_by_kind.csv")
    err_v = df[df.group != "control_both_correct"].bright_lost.dropna()
    ctl_v = df[df.group == "control_both_correct"].bright_lost.dropna()
    p = mannwhitneyu(err_v, ctl_v, alternative="greater").pvalue if len(err_v) and len(ctl_v) else None
    result = {"n_new_idle_confusion": int(len(err_v)), "n_control": int(len(ctl_v)),
              "median_err": float(err_v.median()), "median_control": float(ctl_v.median()),
              "share_over_5pct_err": float((err_v > 0.05).mean()), "share_over_5pct_control": float((ctl_v > 0.05).mean()),
              "mannwhitney_p_err_greater": p,
              "by_case_counts": df[df.group != "control_both_correct"].groupby("case_id").size().to_dict(),
              "note": "時点は1秒ごとで隣どうしが似ているため、p値は独立性の仮定を満たさず参考値"}
    json.dump(result, open(HERE / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 目視用: 取り違え（端の明るさの割合が大きい順と、ランダム）を並べる
    def tiles_for(sub, n):
        out = []
        for i, r in sub.head(n).iterrows():
            im = crops[i].copy()
            im[:, :X0] //= 3; im[:, X1:] //= 3; im[:Y0, X0:X1] //= 3; im[Y1:, X0:X1] //= 3
            cv2.rectangle(im, (X0, Y0), (X1, Y1), (0, 255, 0), 2)
            cv2.putText(im, f"{r.case_id} t={r.t:.0f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(im, f"true {r['true']} / new {r.new}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            out.append(im)
        while len(out) % 4:
            out.append(np.zeros_like(out[0]))
        return np.vstack([np.hstack(out[k:k + 4]) for k in range(0, len(out), 4)])
    errs = df[df.group != "control_both_correct"]
    cv2.imwrite(str(HERE / "plots" / "random_new_idle_confusions.png"),
                tiles_for(errs.sample(frac=1, random_state=0), 16))
    cv2.imwrite(str(HERE / "plots" / "top_edge_new_idle_confusions.png"),
                tiles_for(errs.sort_values("bright_lost", ascending=False).drop_duplicates("case_id"), 12))
    print(summ.to_string())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
