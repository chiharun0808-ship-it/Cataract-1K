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

from dataset_index.prepare_dataset import SurgicalPhaseDataset
from models.cnn_rnn_hybrids import create_cnn_rnn_model

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
DEFAULT_CSV_PATH = "dataset_index_split.csv"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL_NAME = "resnet50_lstm"


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
    }


def main():
    parser = argparse.ArgumentParser(description="best_model.pth をtest splitで評価する")
    parser.add_argument("checkpoint_dir", help="train2.py が保存した result/.../checkpoints_.../ ディレクトリ")
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    eval_config = load_eval_config(checkpoint_dir)
    model_name = eval_config["model_name"]
    include_idle_phase = eval_config["include_idle_phase"]
    num_classes = eval_config["num_classes"]
    csv_path = eval_config["csv_path"]
    batch_size = eval_config["batch_size"]
    active_phases = phases if include_idle_phase else phases_12

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = SurgicalPhaseDataset(csv_path, split='test', exclude_idle=not include_idle_phase)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # load_state_dict で重みを上書きするため、ここでの pretrained_cnn=True は不要
    model = create_cnn_rnn_model(model_name, num_classes=num_classes, pretrained_cnn=False).to(device)
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
