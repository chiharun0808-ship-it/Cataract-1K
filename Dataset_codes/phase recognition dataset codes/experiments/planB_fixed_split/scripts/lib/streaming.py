"""
案B のストリーミング推論で、30枚の入力を作る部品。

91.4% を出した既存の評価（attention_analysis/scripts/streaming_attention_eval.py）と同じ手順で、
前処理だけを案B のもの（lib/preprocess.py）に差し替える。既存のファイルは変更していない。
- 元動画を cv2 で先頭から1枚ずつ読む
- 毎フレーム前処理して deque(maxlen=(30-1)*frame_interval+1) に入れる
- stride_frames ごとに sample_clip_from_buffer(buffer, 30, frame_interval) で窓を取る
  （末尾が今のフレーム。動画冒頭は一番古いフレームを繰り返して埋める）
"""
import sys
from collections import deque

import cv2

from .paths import CODES_DIR

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))
from inference.visualize_phase_on_video import sample_clip_from_buffer  # noqa: E402  既存と同じ窓の取り方
from utils.gt_utils import native_frame_interval  # noqa: E402


def iter_stream_clips(video_path, preprocess, sequence_length=30, stride_sec=1.0, stop_after_frame=None,
                      only_frames=None):
    """(frame_idx, t_sec, clip(T,3,224,224)) を順に返す。
    only_frames を渡すと、その末尾フレームの窓だけを返す（入力一致の確認用）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = native_frame_interval(fps)
    stride = max(1, round(fps * stride_sec))
    buffer = deque(maxlen=(sequence_length - 1) * interval + 1)
    fidx = 0
    try:
        while True:
            if stop_after_frame is not None and fidx > stop_after_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            buffer.append(preprocess(frame))
            emit = (fidx in only_frames) if only_frames is not None else (fidx % stride == 0)
            if emit:
                yield fidx, fidx / fps, sample_clip_from_buffer(buffer, sequence_length, interval)
            fidx += 1
    finally:
        cap.release()


def video_fps_and_interval(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps, native_frame_interval(fps)
