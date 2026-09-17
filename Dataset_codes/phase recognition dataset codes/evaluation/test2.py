"""
学習済みモデルを test split で評価するスクリプト。

train2.py が保存した best_model.pth は val F1 に基づいて選ばれており、
val split は early stopping / モデル選択にも使われている。そのため val F1 を
最終結果として扱うと、モデル選択に使った指標をそのまま報告することになり
楽観的なバイアスがかかる。ここでは学習・モデル選択に一度も使っていない
test split（create_index.py が作成）に対して独立に評価する。

使い方:
    python test2.py <checkpoint_dir>
    例: python test2.py result/epochs_20/checkpoints_0728_12_26

train2.py が checkpoint_dir に書き出す config.json（INCLUDE_IDLE_PHASE・num_classes 等）
を自動的に読み込んで評価設定を合わせる。config.json が無い古いチェックポイント
（例: result/checkpoints_epoch100_..._no_idle など）を評価する場合のみ、
下の DEFAULT_INCLUDE_IDLE_PHASE を手動でそのチェックポイントの学習設定に合わせること。
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataset_index.prepare_dataset import SurgicalPhaseDataset
from models.cnn_rnn_hybrids import create_cnn_rnn_model, CNN_RNN_MODELS
from models.timesformer import create_timesformer_model, TIMESFORMER_MODELS

phases = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle"
]
phases_12 = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics"
]

# config.json が見つからない場合のみ使うフォールバック設定
DEFAULT_INCLUDE_IDLE_PHASE = True
DEFAULT_CSV_PATH = "data/dataset_index_split.csv"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL_NAME = "resnet50_lstm"
# train2.py の SurgicalPhaseDataset のデフォルトと同じ値。TimeSformerはnum_framesが
# 学習時と一致していないと評価できないため、config.jsonに記録されていない古い
# チェックポイントを評価する場合のみこのフォールバック値が使われる。
DEFAULT_SEQUENCE_LENGTH = 30


def load_eval_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"{config_path} を読み込みました: {config}")
        return {
            "model_name": config.get("model_name", DEFAULT_MODEL_NAME),
            "include_idle_phase": config["include_idle_phase"],
            "num_classes": config["num_classes"],
            "csv_path": config["csv_path"],
            "batch_size": config["batch_size"],
            "sequence_length": config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH),
        }

    print(
        f"警告: {config_path} が見つかりません。このスクリプト上部の手動設定 "
        f"(DEFAULT_INCLUDE_IDLE_PHASE={DEFAULT_INCLUDE_IDLE_PHASE}, "
        f"DEFAULT_MODEL_NAME={DEFAULT_MODEL_NAME}) を使用します。"
        f"評価対象のチェックポイントを学習したときの設定と食い違っていないか確認してください。"
    )
    return {
        "model_name": DEFAULT_MODEL_NAME,
        "include_idle_phase": DEFAULT_INCLUDE_IDLE_PHASE,
        "num_classes": 13 if DEFAULT_INCLUDE_IDLE_PHASE else 12,
        "csv_path": DEFAULT_CSV_PATH,
        "batch_size": DEFAULT_BATCH_SIZE,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
    }


def main():
    parser = argparse.ArgumentParser(description="best_model.pth をtest splitで評価する")
    parser.add_argument("checkpoint_dir", help="train2.py が保存した result/.../checkpoints_.../ ディレクトリ")
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument(
        "--csv-path", default=None,
        help="config.jsonのcsv_pathを上書きする（例: train3.pyのチェックポイントを、"
             "学習に使ったスライディングウィンドウ用CSVではなく元のdataset_index_split.csv "
             "のtest splitで評価し、過去のtrain2.py系チェックポイントと同じ土台で比較したい場合）",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    eval_config = load_eval_config(checkpoint_dir)
    model_name = eval_config["model_name"]
    include_idle_phase = eval_config["include_idle_phase"]
    num_classes = eval_config["num_classes"]
    csv_path = args.csv_path or eval_config["csv_path"]
    batch_size = eval_config["batch_size"]
    sequence_length = eval_config["sequence_length"]
    active_phases = phases if include_idle_phase else phases_12

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = SurgicalPhaseDataset(
        csv_path, split='test', sequence_length=sequence_length, exclude_idle=not include_idle_phase
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # load_state_dict で重みを上書きするため、ここでの pretrained_cnn=True は不要。
    # CNN-RNNハイブリッド系とTimeSformer系はそれぞれ別モジュールのレジストリで
    # 管理しているため、model_nameがどちらに属するかでファクトリ関数を切り替える。
    if model_name in CNN_RNN_MODELS:
        model = create_cnn_rnn_model(model_name, num_classes=num_classes, pretrained_cnn=False).to(device)
    elif model_name in TIMESFORMER_MODELS:
        model = create_timesformer_model(
            model_name, num_classes=num_classes, num_frames=sequence_length, pretrained_cnn=False
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"[Test] クリップ数: {len(test_dataset)} | Test F1 (macro): {test_f1:.4f}")

    log_path = checkpoint_dir / "test_log.txt"
    with open(log_path, "w") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"include_idle_phase: {include_idle_phase}\n")
        f.write(f"test_clips: {len(test_dataset)}\n")
        f.write(f"test_f1_macro: {test_f1:.4f}\n")
    print(f"結果を {log_path} に保存しました。")

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # --- フェーズ（クラス）ごとの Accuracy / F1 の算出・保存 ---
    # train2.py と同じ定義: Accuracy は混同行列の対角成分 / その行の合計（= recall）。
    # supportが0のフェーズ（testに出現しないクラス）はaccuracyを0として扱う。
    per_phase_f1 = f1_score(all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0)
    phase_metrics = {}
    for i, phase_name in enumerate(active_phases):
        support = int(cm[i].sum())
        phase_acc = cm[i, i] / support if support > 0 else 0.0
        phase_metrics[phase_name] = {
            "accuracy": float(phase_acc),
            "f1": float(per_phase_f1[i]),
            "support": support,
        }
    phase_metrics_path = checkpoint_dir / "test_phase_metrics.json"
    with open(phase_metrics_path, "w", encoding="utf-8") as f:
        json.dump(phase_metrics, f, indent=2, ensure_ascii=False)

    print("[Test] フェーズごとの Accuracy / F1:")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n--- フェーズごとの Accuracy / F1 ---\n")
        for phase_name, m in phase_metrics.items():
            line = f"  {phase_name}: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f} (support={m['support']})"
            print(line)
            f.write(line + "\n")
    print(f"フェーズごとの指標を {phase_metrics_path} に保存しました。")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=active_phases, yticklabels=active_phases)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Test Confusion Matrix - {model_name} ({num_classes} Classes{"" if include_idle_phase else " without idle"})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_filename = "test_confusion_matrix.png" if include_idle_phase else "test_confusion_matrix_no_idle.png"
    cm_path = checkpoint_dir / cm_filename
    plt.savefig(cm_path)
    plt.close()
    print(f"混同行列を {cm_path} に保存しました。")


if __name__ == "__main__":
    main()
