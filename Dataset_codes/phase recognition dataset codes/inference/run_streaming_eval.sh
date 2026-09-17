#!/bin/bash
# 新チェックポイント(sliding_window)をtest split全11症例の動画に対してストリーミング評価する。
# visualize_phase_on_video.py --gt-csv を各症例動画に対して実行し、pred_log.csv を集める。
# 実行はリポジトリの "phase recognition dataset codes" ディレクトリ直下から:
#   bash inference/run_streaming_eval.sh
set -e

CHECKPOINT_DIR="result/epochs_35/sliding_window_checkpoints_0914_10_46"
OUT_DIR="$CHECKPOINT_DIR/streaming_eval"
VIDEO_DIR="../../Phase_recognition_dataset/dataset_videos/phase_recognition"
ANN_DIR="../../Phase_recognition_dataset/phase_recognition_annotations"

CASES="case_4697 case_4709 case_4811 case_4852 case_4874 case_4924 case_5015 case_5063 case_5300 case_5315 case_5331"

for case in $CASES; do
  echo "=== $case ==="
  python inference/visualize_phase_on_video.py \
    "$CHECKPOINT_DIR" \
    "$VIDEO_DIR/$case.mp4" \
    --stride-sec 1.0 \
    --smooth-window 1 \
    --gt-csv "$ANN_DIR/$case/${case}_annotations_phases.csv" \
    --output "$OUT_DIR/${case}_overlay.mp4"
done

echo "=== ALL DONE ==="
