"""
切り出し前の手術動画(1本まるごと)とアノテーションCSVから、スライディングウィンドウ方式の
学習用クリップを大量に切り出し、SurgicalPhaseDataset(dataset_index/prepare_dataset.py)が
そのまま読める形式のインデックスCSVを作る前処理スクリプト。

## 背景
train2.py の学習は「あらかじめ1フェーズ1クリップに切り出された動画」を使っているが、
visualize_phase_on_video.py による推論は「切り出し前の動画に対するスライディングウィンドウ」
で行っている。この学習時と推論時の入力分布のズレが、フェーズ境界での誤認識やチラつきの
一因になっていると考えられる。本スクリプトは、推論時と同じスライディングウィンドウの
切り出しロジック（gt_utils.py, visualize_phase_on_video.py の sample_clip_from_buffer と同じ
考え方）を学習データ生成にも使うことで、この分布のズレを減らすことを目的とする。

## 方針（ユーザーと合意した内容）
- 境界ラベル: ウィンドウの末尾フレーム時点のフェーズを正解ラベルとする（causal、推論時と同一）
- 抽出間隔: stride_sec=2.5秒（0.5〜1秒だと動画1本あたり数百〜千クリップになり冗長・低速なため、
  中間の値を採用）
- sequence_length・frame_interval（実時間換算）は既存のtrain2.py/SurgicalPhaseDatasetと同じ
  （30フレーム、1/3秒間隔=実時間10秒の窓）を維持し、学習方式の違いの効果だけを切り分ける
- train/val/test split と fold は、既存の dataset_index_split.csv の case_id 単位の割り当てを
  そのまま再利用する（新たに分割し直すとリークや既存結果との不整合が生じるため）

## 出力されるクリップの中身（重要）
各クリップは「学習時と同じ1/3秒間隔で、visualize_phase_on_video.py の
sample_clip_from_buffer と全く同じロジックで既にサンプリング済みの sequence_length(30)枚」を
そのまま30fpsの動画として書き出したものである。つまり SurgicalPhaseDataset 側で再度
フレーム間引きをする必要はなく、常に frame_interval=1 で読めば良い（train3.py 参照）。
動画冒頭でバッファがまだ埋まっていない区間は、visualize_phase_on_video.py の推論時と同じく
一番古いフレームを繰り返して埋める（＝推論時のパディング挙動を学習時にも再現する）。

使い方:
    python build_sliding_window_index.py [--cases case_4697 case_4693 ...] [--stride-sec 2.5]
    例（全54症例を対象に既定値で実行）:
    python build_sliding_window_index.py
    例（動作確認用に2症例だけで試す）:
    python build_sliding_window_index.py --cases case_4697 case_4693
"""
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import pandas as pd

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from utils.gt_utils import native_frame_interval, load_ground_truth_intervals, true_phase_at

PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle"
]
LABEL_TO_IDX = {phase: i for i, phase in enumerate(PHASES)}

SEQUENCE_LENGTH = 30  # train2.py と同じ。フレーム数の最適化は別実験として切り分ける
OUTPUT_FPS = 30  # 書き出すクリップのfps。学習側(frame_interval=1)の前提と合わせる

VIDEO_DIR_TEMPLATE = "../../Phase_recognition_dataset/dataset_videos/phase_recognition/{case}.mp4"
GT_CSV_TEMPLATE = "../../Phase_recognition_dataset/phase_recognition_annotations/{case}/{case}_annotations_phases.csv"

DEFAULT_SPLIT_CSV = "data/dataset_index_split.csv"
DEFAULT_OUTPUT_DIR = Path("sliding_window_dataset")
DEFAULT_OUTPUT_INDEX = "data/dataset_index_sliding_window.csv"


def load_case_split_fold_map(split_csv: str) -> dict:
    """既存の dataset_index_split.csv から case_id -> (split, fold) を復元する。
    1つの case_id に紐づく split/fold は常に同じ値のはず（case単位でGroupShuffleSplit/
    GroupKFoldされているため）なので、最初の行の値をそのまま使う。"""
    df = pd.read_csv(split_csv)
    mapping = {}
    for case_id, group in df.groupby("case_id"):
        splits = group["split"].unique()
        folds = group["fold"].unique()
        if len(splits) != 1:
            raise ValueError(f"{case_id} の split が複数存在します: {splits}")
        mapping[case_id] = {"split": splits[0], "fold": int(folds[0]) if len(folds) == 1 else -1}
    return mapping


def sample_clip_indices(buffer_len: int, sequence_length: int, frame_interval: int):
    """visualize_phase_on_video.py の sample_clip_from_buffer と同じ因果的サンプリング。
    バッファ内での相対インデックス(0-origin, 末尾がbuffer_len-1)のリストを返す。"""
    last_idx = buffer_len - 1
    return [max(last_idx - k * frame_interval, 0) for k in range(sequence_length - 1, -1, -1)]


def extract_windows_for_case(case_id: str, gt_csv_path: str, video_path: str,
                              stride_sec: float, output_dir: Path):
    """1症例の動画を1回だけ順再生しながら、stride_secごとにスライディングウィンドウの
    クリップを切り出す。返り値は [(clip_path, label, t_sec), ...]。"""
    gt_intervals = load_ground_truth_intervals(gt_csv_path, PHASES)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = native_frame_interval(fps)
    stride_frames = max(1, round(fps * stride_sec))
    buffer_maxlen = (SEQUENCE_LENGTH - 1) * frame_interval + 1

    print(f"  {case_id}: fps={fps:.2f} frame_interval={frame_interval} stride_frames={stride_frames}")

    buffer = deque(maxlen=buffer_maxlen)
    rows = []
    frame_idx = 0
    n_written = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        buffer.append(frame_bgr)

        if frame_idx % stride_frames == 0:
            t_sec = frame_idx / fps
            label = true_phase_at(gt_intervals, t_sec)
            label_id = LABEL_TO_IDX[label]

            indices = sample_clip_indices(len(buffer), SEQUENCE_LENGTH, frame_interval)
            clip_dir = output_dir / label
            clip_dir.mkdir(parents=True, exist_ok=True)
            clip_path = clip_dir / f"{case_id}_t{t_sec:07.2f}.mp4"

            writer = cv2.VideoWriter(
                str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (width, height)
            )
            for i in indices:
                writer.write(buffer[i])
            writer.release()

            rows.append({
                "file_path": str(clip_path).replace("\\", "/"),
                "label": label,
                "label_id": label_id,
                "case_id": case_id,
                "source_time_sec": round(t_sec, 2),
            })
            n_written += 1

        frame_idx += 1

    cap.release()
    print(f"  {case_id}: {n_written} 個のウィンドウクリップを書き出しました。")
    return rows


def main():
    parser = argparse.ArgumentParser(description="スライディングウィンドウ方式の学習用クリップとインデックスCSVを作る")
    parser.add_argument("--cases", nargs="*", default=None, help="対象症例を限定する場合（例: case_4697 case_4693）。省略時は全症例")
    parser.add_argument("--stride-sec", type=float, default=2.5, help="ウィンドウを何秒おきに抽出するか")
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV, help="case_idごとのsplit/foldを引き継ぐ既存CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="クリップの出力先ディレクトリ")
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT_INDEX, help="出力するインデックスCSVのパス")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_split_fold = load_case_split_fold_map(args.split_csv)
    target_cases = args.cases if args.cases else sorted(case_split_fold.keys())

    print(f"対象症例数: {len(target_cases)} 本 | stride_sec={args.stride_sec}")

    all_rows = []
    for i, case_id in enumerate(target_cases, 1):
        if case_id not in case_split_fold:
            print(f"警告: {case_id} は {args.split_csv} に見つかりません。スキップします。")
            continue
        print(f"[{i}/{len(target_cases)}] {case_id} を処理中...")
        gt_csv_path = GT_CSV_TEMPLATE.format(case=case_id)
        video_path = VIDEO_DIR_TEMPLATE.format(case=case_id)
        rows = extract_windows_for_case(case_id, gt_csv_path, video_path, args.stride_sec, output_dir)
        split_fold = case_split_fold[case_id]
        for row in rows:
            row["split"] = split_fold["split"]
            row["fold"] = split_fold["fold"]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "file_path", "label", "label_id", "case_id", "split", "fold", "source_time_sec"
    ])
    df.to_csv(args.output_index, index=False)

    print(f"\n完了。{len(df)} 個のウィンドウを {args.output_index} に記録しました。")
    print("split別クリップ数:")
    print(df.groupby("split").size())
    print("split別・症例数:")
    print(df.groupby("split")["case_id"].nunique())
    print("フェーズ別クリップ数:")
    print(df.groupby("label").size().reindex(PHASES))


if __name__ == "__main__":
    main()
