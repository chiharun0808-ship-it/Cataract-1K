# [現在未使用・参考用] import が実フォルダ構成 (dataset_index/prepare_dataset.py) と
# 一致しておらず、このままでは動作しない。check_labels.py も参照。
from prepare_dataset import SurgicalPhaseDataset
val_dataset = SurgicalPhaseDataset("dataset_index_split.csv", split='val')
print(f"検証データ件数: {len(val_dataset)}")
if len(val_dataset) > 0:
    print("検証データの読み込み確認OK")
else:
    print("エラー: 検証データが0件です。split設定を確認してください。")