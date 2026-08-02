import os
import csv
import re
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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

# 交差検証用の fold 数。train2.py が fixed-split / 交差検証のどちらでも同じ
# dataset_index_split.csv を使えるよう、split列(train/val/test)とは別に
# fold列を持たせる。
NUM_FOLDS = 5

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

# 3. 交差検証用の fold 割り当て
# 固定 test set（モデル選択・fold比較のいずれにも使わないホールドアウト）は
# どの fold にも属させない（fold=-1）。test以外の症例（train+val, 症例単位）を
# GroupKFold で NUM_FOLDS 個に分割する。train2.py は交差検証モードのとき
# split列ではなくこの fold列を見て、fold==k を val、fold!=k（かつ test以外）を
# train として扱う。
print(f"\n交差検証用 fold を割り当て中（{NUM_FOLDS}-fold）...")
cv_pool = df[df['split'] != 'test'].copy()
gkf = GroupKFold(n_splits=NUM_FOLDS)
df['fold'] = -1
for fold_idx, (_, val_idx) in enumerate(gkf.split(cv_pool, groups=cv_pool['case_id'])):
    df.loc[cv_pool.index[val_idx], 'fold'] = fold_idx

print(f"fold割り当て結果（症例数）:")
print(df[df['fold'] >= 0].groupby('fold')['case_id'].nunique())

# 4. 結果の保存
df.to_csv(final_csv, index=False)

print(f"--- 完了 ---")
print(f"作成ファイル: {final_csv}")
print("分割結果（症例数）:")
print(df.groupby('split')['case_id'].nunique())

# 検出された症例一覧の確認。以前はここで "../dataset_index.csv" を相対パスで
# 再読み込みしており、output_csv の書き出し先（実行時カレントディレクトリ直下）
# と矛盾していた（カレントディレクトリによっては FileNotFoundError になる）ため、
# 上で読み込み・case_id 付与済みの df をそのまま再利用する形に修正。
unique_cases = sorted(df['case_id'].unique())
print(f"検出されたユニークな症例数: {len(unique_cases)} 本")
print("検出されたケース一覧:", unique_cases)