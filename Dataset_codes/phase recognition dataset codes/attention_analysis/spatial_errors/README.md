# Spatial Attention 解析結果

`analyze_spatial_attention.py` の出力です。既存スクリプト・モデル・Datasetは一切
変更していません（forward hookによる読み取り専用のAttention取得。詳細はスクリプト
冒頭のdocstring参照）。analyze_temporal_attention.py / verify_frame11_positional_bias.py
と対になる解析です。

## 実行条件
- checkpoint: `result\epochs_35\sliding_window_checkpoints_0914_10_46\best_model.pth`
- インデックスCSV: `dataset_index_sliding_window_stride5.csv` (split=test)
- 表示レイヤー: [0, 5, 11]（0-indexed。0=初期層, 11=最終層）
- 対象split内の総ウィンドウ数: 881
- 走査/解析したウィンドウ数: 866 / 74
- only_errors: True

## ディレクトリ構成
```
attention_analysis/spatial_errors/
├── plots/window_XXXX_spatial.png  layer×frameのグリッド。実フレームにヒートマップ重ね表示
├── raw/window_XXXX_spatial_attn.npy  (depth=12, T=30, 14, 14) 全層・全フレームの生データ
└── csv/spatial_attention_summary.csv  1行=1ウィンドウ。表示した(layer,frame)組ごとの
    argmax位置(14x14グリッド内のrow,col)・最大値・集中度(max/mean比)を記録
```

## 可視化対象フレームの選び方
11番目(時間Attention解析で位置バイアスの疑いが強いフレーム)・30番目(query/分類対象
フレーム)・GTフェーズ切り替え直後のフレーム(あれば)を並べて表示します。

## Attentionの解釈について
注意: Attentionが高い = そのフレームが分類の直接的な原因、とは限らない。ここでは「モデルが時間方向のどこを平均的に参照する傾向があるか」を示す。
時間Attention解析(verify_frame11_positional_bias.py)で、Attentionパターンには内容と
無関係な学習済みバイアスが混入し得ることが確認されています。空間Attentionについても、
同様のバイアス(例えば画像中央や特定パッチ位置への固定的な偏り)が無いか、複数ウィンドウを
比較して確認することを推奨します。
