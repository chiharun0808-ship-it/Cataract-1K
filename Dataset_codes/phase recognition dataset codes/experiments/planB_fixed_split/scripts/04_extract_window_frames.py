"""
手順4: 窓に必要なフレームだけを、元動画から PNG（劣化しない形式）で取り出す。
既存結果（91.4%）の再調査専用。案B の本番の入力条件用に流用しない前提。

- 対象は train と val の窓（学習とモデル選択に使う分）。test はストリーミング評価で元動画を直接読むので取り出さない。
- 推論と同じく cv2.VideoCapture で先頭から順に読み、読んだ BGR 配列をそのまま PNG に保存する。
- 保存した PNG をすぐ読み直し、元の配列と全画素が一致することを確かめる（1枚でも違えば終了コード1）。
- 既に同じ PNG があり内容が一致する場合は書き直さない（途中から再開できる）。

出力: frames/repro-baseline/<case>/<フレーム番号6桁>.png
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import FRAMES_DIR, PLAN_DIR, VIDEO_TEMPLATE

WINDOW_CSV = PLAN_DIR / "data" / "windows_repro-baseline_f30_i333ms_s5s.csv"
OUT_ROOT = FRAMES_DIR / "repro-baseline"
SPLITS = ("train", "val")


def frame_path(case_id, idx):
    return OUT_ROOT / case_id / f"{idx:06d}.png"


def main():
    win = pd.read_csv(WINDOW_CSV)
    win = win[win.split.isin(SPLITS)]
    needed = {c: sorted({int(x) for s in g.frame_indices for x in s.split()}) for c, g in win.groupby("case_id")}
    total = sum(len(v) for v in needed.values())
    print(f"対象: {len(needed)} 症例、{total} フレーム（split: {SPLITS}）")

    problems, done = [], 0
    for i, (case, idxs) in enumerate(sorted(needed.items()), 1):
        out_dir = OUT_ROOT / case
        out_dir.mkdir(parents=True, exist_ok=True)
        want = set(idxs)
        last = max(idxs)
        cap = cv2.VideoCapture(VIDEO_TEMPLATE.format(case=case))
        fidx = 0
        while fidx <= last:
            ok, frame = cap.read()
            if not ok:
                problems.append(f"{case}: フレーム {fidx} を読めませんでした（必要な最後のフレームは {last}）")
                break
            if fidx in want:
                p = frame_path(case, fidx)
                if not (p.exists() and np.array_equal(cv2.imread(str(p), cv2.IMREAD_COLOR), frame)):
                    cv2.imwrite(str(p), frame)
                    back = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if back is None or not np.array_equal(back, frame):
                        problems.append(f"{case}: フレーム {fidx} の PNG が元の画素と一致しません")
                done += 1
            fidx += 1
        cap.release()
        print(f"  [{i}/{len(needed)}] {case}: {len(idxs)} 枚")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "既存結果（91.4%）の再調査専用。案B の本番の入力条件用に流用しない",
        "window_csv": WINDOW_CSV.name, "splits": list(SPLITS),
        "n_cases": len(needed), "n_frames": total, "n_saved_and_verified": done,
        "problems": problems,
    }
    json.dump(meta, open(OUT_ROOT / "extract_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if problems or done != total:
        print("[停止] 問題がありました:")
        for p in problems[:30]:
            print("  - " + p)
        sys.exit(1)
    print(f"[OK] {done} 枚を保存し、すべて元の画素と一致することを確認しました。")


if __name__ == "__main__":
    main()
