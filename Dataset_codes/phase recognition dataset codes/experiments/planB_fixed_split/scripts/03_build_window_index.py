"""
手順3: 56症例の学習用の窓の一覧を作る（既存結果の再現専用。案B の本番条件ではない）。

窓の決め方は、現行モデルの学習データを作った data_preparation/build_sliding_window_index.py と
同じ計算にする（画像は保存せず、使うフレーム番号だけを記録する）。
- frame_interval = native_frame_interval(fps)  （1/3秒をフレーム数に換算）
- stride_frames  = max(1, round(fps * 5.0))       （5秒おき）
- frame_idx % stride_frames == 0 のフレームを窓の末尾（query）にする
- 窓のフレーム = 末尾から frame_interval ずつ遡った30枚。動画冒頭は一番古いフレーム（0番）で埋める
- ラベル = 末尾の時刻 t = frame_idx / fps の正解フェーズ（true_phase_at）

確認（失敗したら書き出さずに終了コード1）:
1. 分割は data/case_split_56cases.csv から取り、56症例すべてに split がある。
2. 既存54症例の窓の時刻・ラベル・split が、現行モデルの学習データ
   （data/dataset_index_sliding_window_stride5.csv）と1件残らず一致する。
3. 追加の2症例が train になっている。

フレーム数は CAP_PROP_FRAME_COUNT ではなく、既存スクリプトと同じく実際に読める枚数を数える。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import (BASELINE_WINDOW_CSV, CASE_SPLIT_CSV, CODES_DIR, CONDITIONS_DIR,
                       GT_CSV_TEMPLATE, PLAN_DIR, VIDEO_TEMPLATE)

sys.path.insert(0, str(CODES_DIR))
from utils.gt_utils import load_ground_truth_intervals, native_frame_interval, true_phase_at  # noqa: E402

CONDITION_TAG = "repro-baseline_f30_i333ms_s5s_aa-off"
PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle",
]
LABEL_TO_IDX = {p: i for i, p in enumerate(PHASES)}
OUTPUT_CSV = PLAN_DIR / "data" / "windows_repro-baseline_f30_i333ms_s5s.csv"


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return fps, n


def windows_for_case(case_id, cond):
    seq_len = cond["input"]["sequence_length"]
    stride_sec = cond["input"]["train_window_stride_sec"]
    fps, n_frames = count_frames(VIDEO_TEMPLATE.format(case=case_id))
    interval = native_frame_interval(fps)
    stride = max(1, round(fps * stride_sec))
    gt = load_ground_truth_intervals(GT_CSV_TEMPLATE.format(case=case_id), PHASES)
    rows = []
    for q in range(0, n_frames, stride):
        t = q / fps
        label = true_phase_at(gt, t)
        frames = [max(q - k * interval, 0) for k in range(seq_len - 1, -1, -1)]
        rows.append({"case_id": case_id, "query_frame_idx": q, "source_time_sec": round(t, 2),
                     "label": label, "label_id": LABEL_TO_IDX[label], "fps": fps,
                     "frame_interval": interval, "frame_indices": " ".join(map(str, frames))})
    return rows, fps, n_frames


def fail(msgs):
    print("\n[停止] 確認に失敗したため、書き出していません。")
    for m in msgs[:50]:
        print("  - " + m)
    if len(msgs) > 50:
        print(f"  ...ほか {len(msgs) - 50} 件")
    sys.exit(1)


def main():
    cond = yaml.safe_load(open(CONDITIONS_DIR / f"{CONDITION_TAG}.yaml", encoding="utf-8"))
    split = pd.read_csv(CASE_SPLIT_CSV)
    split_map = dict(zip(split.case_id, split.split))
    if len(split_map) != 56:
        fail([f"分割ファイルの症例数が56ではありません: {len(split_map)}"])
    print("[確認1] 分割ファイルに56症例 → OK")

    all_rows, video_info = [], {}
    for i, case in enumerate(sorted(split_map), 1):
        rows, fps, n = windows_for_case(case, cond)
        for r in rows:
            r["split"] = split_map[case]
        all_rows.extend(rows)
        video_info[case] = {"fps": fps, "n_frames": n, "n_windows": len(rows)}
        print(f"  [{i}/56] {case}: fps={fps:.3f} frames={n} windows={len(rows)}")
    df = pd.DataFrame(all_rows)

    # 確認2: 既存54症例と照合
    old = pd.read_csv(BASELINE_WINDOW_CSV)
    old_key = old.assign(t=old.source_time_sec.round(2))[["case_id", "t", "label", "split"]]
    new54 = df[df.case_id.isin(set(old.case_id))].assign(t=lambda d: d.source_time_sec.round(2))
    merged = old_key.merge(new54[["case_id", "t", "label", "split"]], on=["case_id", "t"],
                           how="outer", suffixes=("_old", "_new"), indicator=True)
    problems = []
    for _, r in merged[merged._merge != "both"].iterrows():
        where = "既存のみ" if r._merge == "left_only" else "新しい一覧のみ"
        problems.append(f"{r.case_id} t={r.t}: 窓が{where}にある（原因: 窓の時刻の計算またはフレーム数の違い）")
    both = merged[merged._merge == "both"]
    for _, r in both[both.label_old != both.label_new].iterrows():
        problems.append(f"{r.case_id} t={r.t}: ラベル 既存={r.label_old} 新={r.label_new}（原因: 正解の求め方の違い）")
    for _, r in both[both.split_old != both.split_new].iterrows():
        problems.append(f"{r.case_id} t={r.t}: split 既存={r.split_old} 新={r.split_new}（原因: 分割ファイルの違い）")
    if problems:
        fail(problems)
    print(f"[確認2] 既存54症例の窓 {len(old)} 件が、時刻・ラベル・split すべて一致 → OK")

    added = df[~df.case_id.isin(set(old.case_id))]
    if sorted(added.case_id.unique()) != ["case_5353", "case_5357"] or set(added.split) != {"train"}:
        fail([f"追加症例の内容が想定と違います: {sorted(added.case_id.unique())}, split={set(added.split)}"])
    print(f"[確認3] 追加2症例の窓 {len(added)} 件はすべて train → OK")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "split", "query_frame_idx", "source_time_sec", "label", "label_id",
            "fps", "frame_interval", "frame_indices"]
    df[cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "condition_tag": CONDITION_TAG,
        "scope": "既存結果（91.4%）の再調査専用。案B の本番の入力条件ではない",
        "n_windows_by_split": df.split.value_counts().to_dict(),
        "n_windows_by_label_train": df[df.split == "train"].label.value_counts().to_dict(),
        "n_unique_frames": int(sum(len(set(" ".join(g.frame_indices).split())) for _, g in df.groupby("case_id"))),
        "videos": video_info,
        "checks_passed": [1, 2, 3],
    }
    json.dump(meta, open(OUTPUT_CSV.with_suffix(".meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n完了: {OUTPUT_CSV.name}（{len(df)} 窓、取り出すフレーム {meta['n_unique_frames']} 枚）")
    print(df.split.value_counts().to_string())


if __name__ == "__main__":
    main()
