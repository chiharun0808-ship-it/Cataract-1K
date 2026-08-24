import gc
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from dataset_index.prepare_dataset import SurgicalPhaseDataset
from models.cnn_rnn_hybrids import create_cnn_rnn_model
from dataset_weight.class_weight import get_class_weights as get_class_weights_13
from dataset_weight.class_weight_without_idle import get_class_weights as get_class_weights_12
from pathlib import Path
from datetime import datetime

#from create_index import phases

phases = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle"
]

# idleフェーズを除外した12クラス版。どちらを使うかは下の INCLUDE_IDLE_PHASE フラグで切り替える
phases_12 = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics"
]

# --- 1. 設定項目 ---
# idleフェーズ(13番目のクラス)を分類に含めるかどうかはこのフラグ一箇所だけで切り替える。
# num_classes・使用するクラス重み・phasesラベル・データセットのidle除外・混同行列の
# クラス数は、すべてここから機械的に導出する（以前は各所を個別にコメントアウトして
# 切り替えていたため、切り替え漏れで実験結果が意図と食い違う原因になっていた）。
INCLUDE_IDLE_PHASE = True
num_classes = 13 if INCLUDE_IDLE_PHASE else 12
active_phases = phases if INCLUDE_IDLE_PHASE else phases_12

# 交差検証を行うかどうかもこのフラグ一箇所だけで切り替える。
# False: 従来通り dataset_index_split.csv の split列(train/val/test)による
#        固定分割で1回だけ学習する（デフォルト、既存の挙動と完全互換）。
# True:  dataset_index/create_index.py が付与した fold列を使い、NUM_FOLDS個の
#        foldそれぞれで学習し、fold間の val F1 の平均・標準偏差を集計する。
#        どちらのモードでも test split には触れない（test2.py で別途評価する）。
USE_CROSS_VALIDATION = True
NUM_FOLDS = 5

# 学習するモデルをここで指定する。複数指定すると、その全モデルを順番に学習・比較できる
# （fixed splitモードならモデルごとに1回、交差検証モードならモデル×foldの全組み合わせ）。
# 指定できる値は models/cnn_rnn_hybrids.py の CNN_RNN_MODELS を参照:
# "resnet50_lstm", "resnet50_gru", "efficientnetb5_lstm", "efficientnetb5_gru"
# 前処理の解像度は224x224に統一しており(dataset_index/prepare_dataset.py)、
# 本来456x456程度を想定するEfficientNetB5系もこの解像度で学習・比較する。
MODEL_NAMES = ["resnet50_lstm"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_path = "dataset_index_split.csv"
batch_size = 4

# EfficientNetB5系は同じ224x224・batch_size=4でもResNet50系よりはるかにGPUメモリを
# 消費する（12GB GPUで batch_size=4, sequence_length=30 だと cuDNN のワークスペースが
# 肥大化しOOMになることを実機で確認済み。batch_size=1なら安定動作する）。
# モデルごとに実際に使うbatch_sizeをここで上書きできるようにし、上の batch_size は
# 「このMODEL_BATCH_SIZE_OVERRIDESに無いモデルのデフォルト値」として扱う。
MODEL_BATCH_SIZE_OVERRIDES = {
    "efficientnetb5_lstm": 1,
    "efficientnetb5_gru": 1,
}

learning_rate = 1e-4
num_epochs = 100
patience = 50

# 現在の日時を取得してフォーマット (例: 7月21日0時49分 -> 0721_0_49)
current_time = datetime.now().strftime("%m%d_%H_%M")


def run_training(save_dir: Path, model_name: str, fold=None) -> float:
    """
    1回分の学習を実行する。
    fold=None のときは split列による固定分割（train/val）で1回学習する。
    fold に整数を渡すと、交差検証モードとしてそのfoldをvalに、残りをtrainとして学習する。
    model_name は models/cnn_rnn_hybrids.py の create_cnn_rnn_model に渡すモデル名。
    戻り値はこの実行で得られた best val F1（macro）。
    """
    os.makedirs(save_dir, exist_ok=True)

    effective_batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, batch_size)

    # この結果フォルダを生成した設定を記録しておく。fold や model の種類を増やすと
    # training_log.txt だけでは「どの設定の結果か」を後から追跡できなくなるため、
    # test2.py もこのファイルを読んで評価時の設定を自動的に合わせられるようにする。
    config = {
        "model_name": model_name,
        "include_idle_phase": INCLUDE_IDLE_PHASE,
        "num_classes": num_classes,
        "csv_path": csv_path,
        "batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "patience": patience,
        "use_cross_validation": USE_CROSS_VALIDATION,
        "fold": fold,
        "num_folds": NUM_FOLDS if USE_CROSS_VALIDATION else None,
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    class_weights = (get_class_weights_13() if INCLUDE_IDLE_PHASE else get_class_weights_12()).to(device)

    # --- 2. データセットの準備 ---
    train_dataset = SurgicalPhaseDataset(csv_path, split='train', fold=fold, exclude_idle=not INCLUDE_IDLE_PHASE)
    val_dataset = SurgicalPhaseDataset(csv_path, split='val', fold=fold, exclude_idle=not INCLUDE_IDLE_PHASE)

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)

    # --- 3. モデル・損失・最適化 ---
    model = create_cnn_rnn_model(model_name, num_classes=num_classes, pretrained_cnn=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. 学習ループ ---
    # 精度や損失を記録してグラフ描画に使用する辞書
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    best_f1 = 0.0
    early_stop_counter = 0
    all_preds, all_labels = [], []

    run_label = f"{model_name}, fold {fold}" if fold is not None else f"{model_name}, fixed split"
    print(f"学習開始 ({run_label}, Device: {device})")

    with open(os.path.join(save_dir, "training_log.txt"), "w", buffering=1) as f:
        for epoch in range(num_epochs):
            # --- 学習フェーズ ---
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            print(f"Epoch {epoch + 1} 開始...")
            all_train_preds, all_train_labels = [], []

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                # 予測値と正解ラベルを蓄積
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if (i + 1) % 50 == 0:
                    print(f"  Batch {i + 1}/{len(train_loader)} 処理中...")

            # --- 検証フェーズ ---
            model.eval()
            all_preds, all_labels = [], []
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_val += labels.size(0)
                    correct_val += (preds == labels).sum().item()

            # --- 履歴への追加とログ記録 ---
            history['train_loss'].append(running_loss / len(train_loader))
            history['val_loss'].append(val_running_loss / len(val_loader))
            history['train_acc'].append(correct_train / total_train)
            history['val_acc'].append(correct_val / total_val)
            # 訓練データと検証データの両方でF1値を算出
            train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)

            log_msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {history['train_loss'][-1]:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            print(log_msg)
            f.write(log_msg + "\n")

            # Early Stopping & Bestモデル保存
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                early_stop_counter = 0
                print("  [Best Model Saved]")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered!")
                    break

    # --- 5. 最終結果出力 ---
    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))

    # LossとAccuracyのグラフを生成・保存
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_result.png'))
    plt.close()

    # 混同行列: val split に出現しないクラスがあっても全クラス分の軸を保持するため
    # labels に num_classes 分を明示的に指定する
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # --- フェーズ（クラス）ごとの Accuracy / F1 の算出・保存 ---
    # Accuracy は混同行列の対角成分 / その行の合計（= recall）として、フェーズごとの
    # 分類性能を評価する。support（そのフェーズの正解サンプル数）が0の場合は0とする。
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
    with open(os.path.join(save_dir, "phase_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(phase_metrics, f, indent=2, ensure_ascii=False)

    print(f"[{run_label}] フェーズごとの Accuracy / F1:")
    with open(os.path.join(save_dir, "training_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n--- フェーズごとの Accuracy / F1 ---\n")
        for phase_name, m in phase_metrics.items():
            line = f"  {phase_name}: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f} (support={m['support']})"
            print(line)
            f.write(line + "\n")

    plt.figure(figsize=(12, 10))  # クラス数が増えるため、少し図のサイズを大きくすると見やすくなります
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=active_phases,
        yticklabels=active_phases
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    title_suffix = "" if INCLUDE_IDLE_PHASE else " without idle"
    fold_suffix = f" - {run_label}" if fold is not None else ""
    plt.title(f'Confusion Matrix ({num_classes} Classes{title_suffix}{fold_suffix})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_filename = 'confusion_matrix.png' if INCLUDE_IDLE_PHASE else 'confusion_matrix_no_idle.png'
    plt.savefig(os.path.join(save_dir, cm_filename))
    plt.close()

    print(f"[{run_label}] 学習終了。best val F1: {best_f1:.4f}（{save_dir} を確認してください）")

    # 複数モデル・複数foldを同一プロセス内で連続して学習する際、GPUメモリを
    # 解放せずに次のモデルへ進むと（特に重いEfficientNetB5系で）CUDA OOMになる。
    # run_training() を抜ける前に明示的に解放する。
    del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_f1


def main():
    # MODEL_NAMES が1つだけの場合は既存の保存先パス（モデル名フォルダなし）を維持し、
    # 複数指定した場合のみモデル名のフォルダを1階層追加して結果を分けて保存する。
    multi_model = len(MODEL_NAMES) > 1

    if USE_CROSS_VALIDATION:
        cv_root = Path("result") / f"epochs_{num_epochs}" / f"cv_{NUM_FOLDS}fold_{current_time}"
        os.makedirs(cv_root, exist_ok=True)

        all_summaries = {}
        for model_name in MODEL_NAMES:
            model_root = (cv_root / model_name) if multi_model else cv_root

            fold_f1s = []
            for fold in range(NUM_FOLDS):
                fold_dir = model_root / f"fold_{fold}"
                best_f1 = run_training(fold_dir, model_name=model_name, fold=fold)
                fold_f1s.append(best_f1)

            summary = {
                "model_name": model_name,
                "num_folds": NUM_FOLDS,
                "fold_best_val_f1": fold_f1s,
                "mean_val_f1": float(np.mean(fold_f1s)),
                "std_val_f1": float(np.std(fold_f1s)),
            }
            with open(model_root / "cv_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[{model_name}] 交差検証終了。mean val F1: {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}")
            all_summaries[model_name] = summary

        if multi_model:
            with open(cv_root / "cv_summary_all_models.json", "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            print(f"全モデルの交差検証結果を {cv_root / 'cv_summary_all_models.json'} にまとめました。")
    else:
        for model_name in MODEL_NAMES:
            base_dir = Path("result") / f"epochs_{num_epochs}" / f"checkpoints_{current_time}"
            save_dir = (base_dir / model_name) if multi_model else base_dir
            run_training(save_dir, model_name=model_name, fold=None)


if __name__ == "__main__":
    main()
