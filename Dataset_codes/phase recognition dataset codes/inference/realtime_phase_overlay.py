"""
学習済みモデルを使い、手術動画全体にフェーズ認識結果をリアルタイム風に重ねて表示する
本番用スクリプト。

visualize_phase_on_video.py との違い:
  visualize_phase_on_video.py は GT アノテーションとの比較・一致率集計・チラつき分析まで
  行う「比較検証用プロトタイプ」。本スクリプトはそれらの分析機能を持たず、
  「チェックポイント + 動画 → フェーズを重ねた動画（および任意でライブプレビュー）」
  だけを行うシンプルな実運用向け版。train2.py・train3.py どちらの学習方式の
  チェックポイントでも、config.json のスキーマが同じなのでそのまま使える。

使い方:
    python realtime_phase_overlay.py <checkpoint_dir> <video_path> \
        [--output out.mp4] [--stride-sec 1.0] [--smooth-window 5] [--display]
    例:
    python realtime_phase_overlay.py result/epochs_100/sliding_window_cv_5fold_XXXX/fold_0 \
        "../../Phase_recognition_dataset/dataset_videos/phase_recognition/case_4709.mp4" \
        --smooth-window 5 --display
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from models.cnn_rnn_hybrids import create_cnn_rnn_model, CNN_RNN_MODELS
from models.timesformer import create_timesformer_model, TIMESFORMER_MODELS
from evaluation.test2 import load_eval_config, phases, phases_12
from utils.gt_utils import native_frame_interval

PHASE_COLORS = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (156, 204, 101),
    (92, 107, 192), (240, 98, 146), (0, 121, 107), (255, 202, 40),
    (120, 120, 120),
]


def build_model(model_name, num_classes, sequence_length, device):
    if model_name in CNN_RNN_MODELS:
        return create_cnn_rnn_model(model_name, num_classes=num_classes, pretrained_cnn=False).to(device)
    elif model_name in TIMESFORMER_MODELS:
        return create_timesformer_model(
            model_name, num_classes=num_classes, num_frames=sequence_length, pretrained_cnn=False
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def preprocess_frame(frame_bgr):
    """cv2で読んだBGRフレームを学習時と同じ前処理（RGB化・0-1正規化・224x224リサイズ）のテンソルにする"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    return tensor


def sample_clip_from_buffer(buffer, sequence_length, frame_interval):
    """バッファされたフレームから、学習時と同じ間隔で過去sequence_length枚を
    サンプリングする（現在フレームが末尾になる、因果的なウィンドウ）。"""
    last_idx = len(buffer) - 1
    indices = [max(last_idx - k * frame_interval, 0) for k in range(sequence_length - 1, -1, -1)]
    return torch.stack([buffer[i] for i in indices])


def draw_phase_label(frame_bgr, phase_name, color):
    label = f"Phase: {phase_name}" if phase_name is not None else "Phase: (認識中...)"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame_bgr, (10, 10), (10 + text_w + 20, 10 + text_h + baseline + 20), color, thickness=-1)
    cv2.putText(
        frame_bgr, label, (20, 10 + text_h + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="手術動画全体にフェーズ認識結果をリアルタイム風に重ねて表示する（本番用）")
    parser.add_argument("checkpoint_dir", help="train2.py / train3.py が保存した result/.../fold_X などのディレクトリ")
    parser.add_argument("video_path", help="切り出し前の手術動画（例: case_XXXX.mp4）へのパス")
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--output", default=None, help="出力動画パス（省略時は入力ファイル名+_phase_overlay.mp4）")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="何秒おきに推論し直すか")
    parser.add_argument("--smooth-window", type=int, default=1, help="直近何回分のsoftmax確率を平均するか（1=平滑化なし）")
    parser.add_argument("--display", action="store_true", help="処理しながらウィンドウにライブプレビューを表示する（'q'キーで中断）")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    video_path = Path(args.video_path)
    output_path = Path(args.output) if args.output else video_path.with_name(video_path.stem + "_phase_overlay.mp4")

    eval_config = load_eval_config(checkpoint_dir)
    model_name = eval_config["model_name"]
    include_idle_phase = eval_config["include_idle_phase"]
    num_classes = eval_config["num_classes"]
    sequence_length = eval_config["sequence_length"]
    active_phases = phases if include_idle_phase else phases_12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes, sequence_length, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = native_frame_interval(fps)
    stride_frames = max(1, round(fps * args.stride_sec))
    buffer_maxlen = (sequence_length - 1) * frame_interval + 1

    print(
        f"動画: {video_path.name} | fps={fps:.2f} | 総フレーム数={total_frames} | "
        f"換算frame_interval={frame_interval} | stride={stride_frames}フレームごと | "
        f"checkpoint={checkpoint_path}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    buffer = deque(maxlen=buffer_maxlen)
    prob_history = deque(maxlen=max(1, args.smooth_window))
    current_phase_name = None
    current_color = (120, 120, 120)
    frame_idx = 0
    t_start = time.time()

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            buffer.append(preprocess_frame(frame_bgr))

            if frame_idx % stride_frames == 0:
                clip = sample_clip_from_buffer(buffer, sequence_length, frame_interval)
                clip = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu()

                prob_history.append(probs)
                avg_probs = torch.stack(list(prob_history)).mean(dim=0)
                pred_idx = int(avg_probs.argmax().item())
                current_phase_name = active_phases[pred_idx]
                current_color = PHASE_COLORS[pred_idx % len(PHASE_COLORS)]

            draw_phase_label(frame_bgr, current_phase_name, current_color)
            writer.write(frame_bgr)

            if args.display:
                cv2.imshow("Phase Overlay (press q to stop)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("ユーザー操作により中断しました。")
                    break

            frame_idx += 1
            if frame_idx % (stride_frames * 30) == 0:
                elapsed = time.time() - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"  処理中... {frame_idx}/{total_frames} フレーム ({proc_fps:.1f} fps処理速度)")

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"完了。出力動画: {output_path} | 処理時間: {elapsed:.1f}秒 ({frame_idx / elapsed:.1f} fps処理速度)")


if __name__ == "__main__":
    main()
