# check_labels.py
import pandas as pd

df = pd.read_csv("../dataset_index_split.csv")
unique_labels = df['label_id'].unique()
print(f"存在するラベルID: {sorted(unique_labels)}")
# 期待: [0, 1, ..., 12]