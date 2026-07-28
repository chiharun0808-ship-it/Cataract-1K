import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 修正点: sep=',' を明示的に指定します
df = pd.read_csv('manifest_copy.csv', sep=',')

# 列名の確認（デバッグ用）
print("--- 修正後に認識された列名一覧 ---")
print(df.columns.tolist())

# 'name'列が存在することを確認してから処理を実行
if 'name' in df.columns:
    # case_id作成
    df['case_id'] = df['name'].str.replace('.mp4', '', regex=False)

    # 1. Train / その他(Val+Test) に分割 (80%)
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.57, random_state=42)
    train_idx, other_idx = next(gss_train.split(df, groups=df['case_id']))

    df.loc[df.index[train_idx], 'split'] = 'train'
    temp_df = df.iloc[other_idx]

    # 2. 残りを Val / Test に分割 (50%ずつ)
    gss_val_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df['case_id']))

    # 分割結果を元のdfに反映
    df.loc[temp_df.iloc[val_idx].index, 'split'] = 'val'
    df.loc[temp_df.iloc[test_idx].index, 'split'] = 'test'

    # 結果確認と保存
    print(df.groupby('split')['case_id'].nunique())
    df.to_csv('manifest_split.csv', index=False, sep=',')
    print("分割完了: 'manifest_split.csv' を保存しました。")
else:
    print("エラー: まだ 'name' 列が見つかりません。")