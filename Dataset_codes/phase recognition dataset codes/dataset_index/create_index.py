import os
import csv
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- 設定項目 ---
dataset_root = r"C:\Users\kit02\GitHub\Cataract-1K\Phase_recognition_dataset\Training_Dataset"
output_csv = "dataset_index.csv"
final_csv = "dataset_index_split.csv"

phases = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle"
]

label_to_idx = {phase: i for i, phase in enumerate(phases)}

# 1. CSV作成処理
print("CSVを作成中...")
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "label", "label_id"])

    for phase in phases:
        phase_dir = os.path.join(dataset_root, phase)
        if os.path.exists(phase_dir):
            for video_file in os.listdir(phase_dir):
                if video_file.endswith(".mp4"):
                    full_path = os.path.join(phase_dir, video_file)
                    writer.writerow([full_path, phase, label_to_idx[phase]])

# 2. 分割処理
print("分割処理を実行中...")
df = pd.read_csv(output_csv)

def extract_case_id(path):
    match = re.search(r'(case_\d+)', path)
    return match.group(1) if match else None

df['case_id'] = df['file_path'].apply(extract_case_id)

# 万が一 case_id が抽出できなかったデータがないか確認
if df['case_id'].isnull().any():
    raise ValueError("エラー: case_id を抽出できなかったファイルパスがあります。正規表現を確認してください。")

# 1段階目: Train の切り出し (例: 60%)
gss_train = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
train_idx, other_idx = next(gss_train.split(df, groups=df['case_id']))

# 確実にインデックスを指定して train を割り当て
df.loc[df.index[train_idx], 'split'] = 'train'

# 残りのデータを確実に抽出
temp_df = df.iloc[other_idx].copy()

# 2段階目: 残りを Val と Test に 50% ずつ分割
gss_val_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df['case_id']))

# 元の dataframe のインデックスに対して直接代入する
df.loc[temp_df.index[val_idx], 'split'] = 'val'
df.loc[temp_df.index[test_idx], 'split'] = 'test'

# 念のため、未割当（NaN）の行がないか最終チェック
unassigned_count = df['split'].isnull().sum()
if unassigned_count > 0:
    print(f"警告: どのグループにも割り当てられなかった行が {unassigned_count} 件あります。")
else:
    print("すべてのデータが正常にいずれかのグループに割り当てられました。")

# 3. 結果の保存
df.to_csv(final_csv, index=False)

print(f"--- 完了 ---")
print(f"作成ファイル: {final_csv}")
print("分割結果（症例数）:")
print(df.groupby('split')['case_id'].nunique())

df = pd.read_csv("../dataset_index.csv")

def extract_case_id(path):
    match = re.search(r'(case_\d+)', path)
    return match.group(1) if match else None

df['case_id'] = df['file_path'].apply(extract_case_id)

unique_cases = sorted(df['case_id'].unique())
print(f"検出されたユニークな症例数: {len(unique_cases)} 本")
print("検出されたケース一覧:", unique_cases)