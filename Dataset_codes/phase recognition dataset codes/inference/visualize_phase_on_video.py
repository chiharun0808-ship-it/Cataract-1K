"""
学習済みモデルを使い、手術動画全体に対してフェーズ認識結果を画面に重ねて表示する
プロトタイプスクリプト。

train2.py / test2.py はあらかじめフェーズごとに切り出された短いクリップ単位で
学習・評価しているが、卒業研究の最終目標は「切り出す前の手術動画1本を入力して、
今どのフェーズかを動画上に表示する」こと。本スクリプトはそのための最初の
プロトタイプとして、切り出し前の動画（例: Phase_recognition_dataset/dataset_videos/
phase_recognition/case_XXXX.mp4）に対してスライディングウィンドウで推論し、
予測フェーズ名をフレームに焼き込んで新しい動画として書き出す。

フェーズ遷移の順序などの外部知識は使わず、モデル自身の出力（softmax確率）のみを
直近 --smooth-window 回分平均してからargmaxを取ることで、正解ラベルが切り替わって
いない区間で数秒だけ別のフェーズに誤って切り替わってしまう「チラつき」を抑える。

--gt-csv で create_index.py 実行前のアノテーションCSV（例:
Phase_recognition_dataset/phase_recognition_annotations/case_XXXX/
case_XXXX_annotations_phases.csv）を渡すと、動画全体に対する時刻ごとの
正解フェーズと比較して、区間ごとの一致率・予測の切り替わり回数（チラつきの目安）を
集計する。

使い方:
    python visualize_phase_on_video.py <checkpoint_dir> <video_path> \
        [--output out.mp4] [--stride-sec 1.0] [--smooth-window 5] [--gt-csv annotations_phases.csv]
    例:
    python visualize_phase_on_video.py result/epochs_100/cv_5fold_0824_17_48/fold_0 \
        "../../Phase_recognition_dataset/dataset_videos/phase_recognition/case_4697.mp4" \
        --smooth-window 5 \
        --gt-csv "../../Phase_recognition_dataset/phase_recognition_annotations/case_4697/case_4697_annotations_phases.csv"
"""
import argparse
import csv
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
from utils.gt_utils import (
    TRAIN_FPS_REFERENCE,
    TRAIN_FRAME_INTERVAL_REFERENCE,
    native_frame_interval,
    normalize_phase_name,
    load_ground_truth_intervals,
    true_phase_at,
)

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
    """
    現在までにバッファされたフレームから、学習時と同じ間隔で過去sequence_length枚を
    サンプリングする（現在フレームが末尾になる、因果的なウィンドウ）。
    バッファがまだ足りない動画冒頭では、一番古いフレームを繰り返して埋める
    （train2.py側の「足りない分は直近フレームで埋める」パディングと対称的な扱い）。
    """
    last_idx = len(buffer) - 1
    indices = [max(last_idx - k * frame_interval, 0) for k in range(sequence_length - 1, -1, -1)]
    return torch.stack([buffer[i] for i in indices])


def summarize_prediction_log(log_rows):
    """(time_sec, pred, true) のログから一致率と予測の切り替わり回数（チラつきの目安）を集計する"""
    total = len(log_rows)
    correct = sum(1 for _, pred, true in log_rows if pred == true)
    pred_switches = sum(1 for i in range(1, total) if log_rows[i][1] != log_rows[i - 1][1])
    true_switches = sum(1 for i in range(1, total) if log_rows[i][2] != log_rows[i - 1][2])
    print(f"[GT比較] 区間数={total} | 一致率={correct/total*100:.1f}% | "
          f"予測の切り替わり回数={pred_switches} (正解の切り替わり回数={true_switches})")
    return {"total": total, "correct": correct, "pred_switches": pred_switches, "true_switches": true_switches}


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
    parser = argparse.ArgumentParser(description="手術動画全体にフェーズ認識結果を重ねて表示する")
    parser.add_argument("checkpoint_dir", help="train2.py が保存した result/.../fold_X などのディレクトリ")
    parser.add_argument("video_path", help="切り出し前の手術動画（例: case_XXXX.mp4）へのパス")
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--output", default=None, help="出力動画パス（省略時は入力ファイル名+_phase_overlay.mp4）")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="何秒おきに推論し直すか")
    parser.add_argument("--smooth-window", type=int, default=1, help="直近何回分のsoftmax確率を平均するか（1=平滑化なし）")
    parser.add_argument("--gt-csv", default=None, help="正解フェーズのアノテーションCSV（比較・集計用、省略可）")
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

    # 学習時のサンプル間隔（実時間換算）をこの動画のfpsに合わせて換算する
    frame_interval = native_frame_interval(fps)
    stride_frames = max(1, round(fps * args.stride_sec))
    buffer_maxlen = (sequence_length - 1) * frame_interval + 1

    print(
        f"動画: {video_path.name} | fps={fps:.2f} | 総フレーム数={total_frames} | "
        f"換算frame_interval={frame_interval} | stride={stride_frames}フレームごと"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    gt_intervals = load_ground_truth_intervals(args.gt_csv, active_phases) if args.gt_csv else None
    prediction_log = []  # (time_sec, pred_phase, true_phase)

    buffer = deque(maxlen=buffer_maxlen)
    prob_history = deque(maxlen=max(1, args.smooth_window))
    current_phase_name = None
    current_color = (120, 120, 120)
    frame_idx = 0

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

                # 直近smooth_window回分のsoftmax確率を平均してからargmaxを取ることで、
                # 1回だけ外れた推論による瞬間的な誤ったフェーズ切り替わりを抑える。
                prob_history.append(probs)
                avg_probs = torch.stack(list(prob_history)).mean(dim=0)
                pred_idx = int(avg_probs.argmax().item())
                current_phase_name = active_phases[pred_idx]
                current_color = PHASE_COLORS[pred_idx % len(PHASE_COLORS)]

                if gt_intervals is not None:
                    t_sec = frame_idx / fps
                    true_phase = true_phase_at(gt_intervals, t_sec)
                    prediction_log.append((t_sec, current_phase_name, true_phase))

            draw_phase_label(frame_bgr, current_phase_name, current_color)
            writer.write(frame_bgr)

            frame_idx += 1
            if frame_idx % (stride_frames * 30) == 0:
                print(f"  処理中... {frame_idx}/{total_frames} フレーム")

    cap.release()
    writer.release()
    print(f"完了。出力動画: {output_path}")

    if gt_intervals is not None:
        summarize_prediction_log(prediction_log)
        log_path = output_path.with_suffix(".pred_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["time_sec", "pred_phase", "true_phase"])
            writer_csv.writerows(prediction_log)
        print(f"予測ログを {log_path} に保存しました。")


if __name__ == "__main__":
    main()
