# SurgCBVR-CAT（白内障手術動画クリップの内容ベース検索ベンチマーク）

- 引用キー：tbd_surgcbvr（論文が出たら付け直す。README の論文欄は「TODO: Data in Brief」で未出版）
- 著者：未確認（GitHub の所有者は mohaEs。README に著者名なし）
- 発表：未出版（Data in Brief に投稿予定とある）
- リンク：https://github.com/mohaEs/SurgCBVR-CAT（MIT ライセンス）。クリップと注釈は Hugging Face の mohaEs/SurgCBVR-CAT
- 状態：未読（論文はまだない）

## 一言で
Cataract-1K の工程ラベル付き 56 本から 6,487 個のクリップを切り出し、「同じ工程のクリップを他の患者の手術から探す」検索の評価基盤にしたもの。特徴は DINOv2・VideoMAE V2・C3D。ラベルは phase（step ではない）。

## 読むときに確かめたいこと
- 論文が出たか（出たら登録し直す）。
