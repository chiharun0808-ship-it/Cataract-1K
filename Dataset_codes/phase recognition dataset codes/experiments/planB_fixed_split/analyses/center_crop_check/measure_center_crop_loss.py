"""
中央切り出し（事前学習と同じ「短い辺を256 → 中央224×224」）で、画面の端から何が失われるかを測る。
読むだけの分析で、学習も推論もしない（2026-10-05、ユーザーの依頼）。

- 対象: train と val の全窓の末尾フレーム（frames/repro-baseline/ の PNG。512×384）
- 切り出しで残る範囲（元画像の座標）: x 88〜424, y 24〜360（336×336）
- 「何かが映っている」の目安を2つ使う
  1. 明るい画素: グレースケール >= 100（眼球、器具、反射など。暗い背景・布と区別する）
  2. 輪郭の画素: Canny(50, 150) の輪郭（器具の縁など）
- それぞれ「全体のうち、切れる部分に入る割合」をフレームごとに計算し、全体・症例・フェーズで集計する
- 割合の大きいフレーム上位を画像にまとめ、目で確かめられるようにする

出力: csv/per_frame.csv, csv/by_phase.csv, csv/by_case.csv, plots/top_loss_frames.png, summary.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from lib.paths import FRAMES_DIR, PLAN_DIR  # noqa: E402

WINDOW_CSV = PLAN_DIR / "data" / "windows_repro-baseline_f30_i333ms_s5s.csv"
FRAME_ROOT = FRAMES_DIR / "repro-baseline"
BRIGHT_T = 100


def keep_box(w, h, resize_short=256, crop=224):
    s = resize_short / min(w, h)
    kw, kh = crop / s, crop / s
    x0, y0 = (w - kw) / 2, (h - kh) / 2
    return int(round(x0)), int(round(y0)), int(round(x0 + kw)), int(round(y0 + kh))


def main():
    win = pd.read_csv(WINDOW_CSV)
    win = win[win.split.isin(["train", "val"])]
    rows = []
    for _, r in win.iterrows():
        q = int(r.frame_indices.split()[-1])
        p = FRAME_ROOT / r.case_id / f"{q:06d}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        x0, y0, x1, y1 = keep_box(w, h)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inside = np.zeros_like(gray, dtype=bool)
        inside[y0:y1, x0:x1] = True
        bright = gray >= BRIGHT_T
        edges = cv2.Canny(gray, 50, 150) > 0
        def lost(mask):
            tot = mask.sum()
            return float((mask & ~inside).sum() / tot) if tot else 0.0
        rows.append({"case_id": r.case_id, "split": r.split, "label": r.label, "query_frame_idx": q,
                     "time_sec": r.source_time_sec, "bright_lost": lost(bright), "edge_lost": lost(edges),
                     "lr_bright_lost": float((bright[:, :x0].sum() + bright[:, x1:].sum()) / max(bright.sum(), 1)),
                     "tb_bright_lost": float((bright[:y0, x0:x1].sum() + bright[y1:, x0:x1].sum()) / max(bright.sum(), 1))})
    df = pd.DataFrame(rows)
    (HERE / "csv").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)
    df.to_csv(HERE / "csv" / "per_frame.csv", index=False)

    def q(s):
        return pd.Series({"n": len(s), "mean": s.mean(), "median": s.median(), "p90": s.quantile(0.9),
                          "max": s.max(), "share_over_10pct": (s > 0.10).mean()})
    by_phase = df.groupby("label")["bright_lost"].apply(q).unstack()
    by_phase.join(df.groupby("label")["edge_lost"].median().rename("edge_median")).round(4).to_csv(HERE / "csv" / "by_phase.csv")
    df.groupby("case_id")["bright_lost"].apply(q).unstack().round(4).to_csv(HERE / "csv" / "by_case.csv")

    summary = {
        "n_frames": len(df), "keep_box_xyxy_on_512x384": keep_box(512, 384),
        "bright_lost": q(df.bright_lost).round(4).to_dict(),
        "edge_lost": q(df.edge_lost).round(4).to_dict(),
        "bright_lost_left_right_median": round(float(df.lr_bright_lost.median()), 4),
        "bright_lost_top_bottom_median": round(float(df.tb_bright_lost.median()), 4),
    }
    json.dump(summary, open(HERE / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 失う割合の大きいフレーム上位12枚（症例が重ならないように選ぶ）
    top = df.sort_values("bright_lost", ascending=False).drop_duplicates("case_id").head(12)
    tiles = []
    for _, r in top.iterrows():
        img = cv2.imread(str(FRAME_ROOT / r.case_id / f"{r.query_frame_idx:06d}.png"))
        x0, y0, x1, y1 = keep_box(img.shape[1], img.shape[0])
        out = img.copy()
        out[:, :x0] //= 3; out[:, x1:] //= 3; out[:y0, x0:x1] //= 3; out[y1:, x0:x1] //= 3
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, f"{r.case_id} {r.label}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(out, f"bright lost {r.bright_lost:.0%}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(out)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(HERE / "plots" / "top_loss_frames.png"), grid)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(by_phase.round(3).to_string())


if __name__ == "__main__":
    main()
