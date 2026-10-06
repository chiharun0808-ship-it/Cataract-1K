"""
手順5: 学習側と推論側の入力が「数値として完全に同じ」かを確かめる。

- 学習側: lib/window_dataset.py の WindowFrameDataset（PNG → preprocess_frame_vit）
- 推論側: lib/streaming.py の iter_stream_clips（元動画を先頭から読み → preprocess_frame_vit →
  既存の評価と同じバッファと sample_clip_from_buffer）。案B の評価はこの部品を使う。
- 対象: train と val の全45症例から、症例ごとに4窓
  （動画の最初の窓＝冒頭を埋める場合、2番目の窓＝一部を埋める場合、ランダムに2窓。乱数の固定値 0）
- 判定: torch.equal（全要素が完全に一致）。1窓でも違えば終了コード1。
- あわせて、前処理後の値の範囲と形を記録する（正規化がかかっていることの確認）。
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import PLAN_DIR, VIDEO_TEMPLATE
from lib.preprocess import PREPROCESS_NAME, describe, preprocess_frame_vit
from lib.streaming import iter_stream_clips
from lib.window_dataset import WindowFrameDataset

WINDOW_CSV = PLAN_DIR / "data" / "windows_repro-baseline_f30_i333ms_s5s.csv"
WINDOWS_PER_CASE = 4


def main():
    rng = random.Random(0)
    datasets = {s: WindowFrameDataset(WINDOW_CSV, s) for s in ("train", "val")}
    by_case = {}
    for split, ds in datasets.items():
        for case, g in ds.data.groupby("case_id"):
            g = g.sort_values("query_frame_idx")
            picks = list(g.index[:2]) + rng.sample(list(g.index[2:]), min(WINDOWS_PER_CASE - 2, len(g) - 2))
            by_case[case] = [(split, i, int(ds.data.loc[i, "query_frame_idx"])) for i in picks]
    n_total = sum(len(v) for v in by_case.values())
    print(f"確認する窓: {n_total}（{len(by_case)} 症例）／前処理: {PREPROCESS_NAME}")

    results, mismatches = [], []
    for n, (case, items) in enumerate(sorted(by_case.items()), 1):
        targets = {q: (s, i) for s, i, q in items}
        for fidx, _, infer in iter_stream_clips(VIDEO_TEMPLATE.format(case=case), preprocess_frame_vit,
                                                stop_after_frame=max(targets), only_frames=set(targets)):
            split, i = targets[fidx]
            train, _ = datasets[split][i]
            equal = torch.equal(infer, train)
            results.append({"case_id": case, "split": split, "query_frame_idx": fidx, "equal": equal,
                            "max_abs_diff": float((infer - train).abs().max()),
                            "shape": "x".join(map(str, train.shape)),
                            "min": float(train.min()), "max": float(train.max()), "mean": float(train.mean())})
            if not equal:
                mismatches.append(f"{case} query={fidx}: 不一致（最大の差 {results[-1]['max_abs_diff']:.3g}）")
        print(f"  [{n}/{len(by_case)}] {case}: {len(items)} 窓")

    df = pd.DataFrame(results)
    out = PLAN_DIR / "data" / "check_train_inference_input_match.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    meta = {"checked_at": datetime.now().isoformat(timespec="seconds"), "preprocess": describe(),
            "n_windows": len(df), "n_equal": int(df.equal.sum()) if len(df) else 0,
            "value_range": [float(df["min"].min()), float(df["max"].max())] if len(df) else None,
            "shape": sorted(df["shape"].unique().tolist()) if len(df) else None, "mismatches": mismatches}
    json.dump(meta, open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"入力の形: {meta['shape']}、値の範囲: {meta['value_range']}（正規化後は負の値を含むはず）")
    if mismatches or len(df) != n_total:
        print("[停止] 一致しない窓がありました:")
        for m in mismatches[:30]:
            print("  - " + m)
        sys.exit(1)
    print(f"[OK] {len(df)} 窓すべてで、学習側と推論側の入力が完全に一致しました。")


if __name__ == "__main__":
    main()
