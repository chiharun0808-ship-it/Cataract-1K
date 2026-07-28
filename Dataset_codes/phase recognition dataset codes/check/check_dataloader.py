from prepare_dataset import SurgicalPhaseDataset
val_dataset = SurgicalPhaseDataset("dataset_index_split.csv", split='val')
print(f"検証データ件数: {len(val_dataset)}")
if len(val_dataset) > 0:
    print("検証データの読み込み確認OK")
else:
    print("エラー: 検証データが0件です。split設定を確認してください。")