import pandas as pd
import torch


def get_class_weights(csv_path='dataset_index_split.csv'):
    df = pd.read_csv(csv_path)

    #trainデータの集計（全クラスID 0-12 が必ず含まれるように補完）
    all_label_ids = list(range(13))
    train_counts = df[df['split'] == 'train']['label_id'].value_counts().reindex(all_label_ids,

                                                                               fill_value=1).sort_index()

    counts_tensor = torch.tensor(train_counts.values, dtype=torch.float)

    # 重み計算 (逆数で算出)
    num_classes = len(counts_tensor)
    weights = counts_tensor.sum() / (num_classes * counts_tensor)

    return weights


if __name__ == "__main__":
    # 直接実行したときは確認表示
    weights = get_class_weights()
    print("--- 算出された重み ---")
    print(weights)