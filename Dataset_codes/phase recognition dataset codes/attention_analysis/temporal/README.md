# Temporal Attention 解析結果

このディレクトリは `analyze_temporal_attention.py` の出力です。
このスクリプトは既存の学習・評価・推論スクリプト（train2.py, train3.py, test2.py,
visualize_phase_on_video.py, realtime_phase_overlay.py, models/timesformer.py,
dataset_index/prepare_dataset.py 等）を一切変更していません。すべて読み取り専用で
import して利用しています（Attention重みの取得も forward hook による読み取り専用の
再計算で、モデルのforward計算自体は変更していません）。

## 実行条件
- checkpoint: `result\epochs_35\sliding_window_checkpoints_0914_10_46\best_model.pth`
- インデックスCSV: `dataset_index_sliding_window_stride5.csv` (split=test)
- 対象split内の総ウィンドウ数: 881
- 走査したウィンドウ数: 881 / 解析（保存）したウィンドウ数: 881
- only_errors: False

## ディレクトリ構成
```
attention_analysis/temporal/
├── plots/window_XXXX.png   1ウィンドウ=1枚。30フレームのTemporal Attention重要度の棒グラフ。
│                           GTフェーズ切り替えがウィンドウ内にある場合のみ境界線を表示。
├── raw/window_XXXX_attn.npy  層別(depth=12)・head平均後・空間パッチ平均後の生データ。
│                             shape=(12, 30, 30) = (layer, query_frame, key_frame)。
│                             CSVのframe_XX_attention列は、これをさらにlayer方向に
│                             平均し、query=30番目(最後のフレーム)の行を取り出したもの。
└── csv/attention_summary.csv  1行=1ウィンドウ。列の意味は下記。
```

## attention_summary.csv の列
- window_id, case_id, video_id, source_case_video, window_clip_path
- window_start_sec, window_end_sec: ウィンドウの実時間範囲(再構成値。下記参照)
- true_label, true_label_id: 正解ラベル(sliding-window CSVのlabel列そのもの。厳密値)
- predicted_label, predicted_label_id, correct, confidence: モデルの予測
- has_transition, transition_frame, n_transitions: GTフェーズ切り替えの有無・位置(1-indexed)・回数
  ※ transition_frameは「切り替え後、最初に新フェーズになるフレーム番号」
  ※ 30フレーム個別の時刻はCSVに保存されていないため、ウィンドウ末尾時刻から
     1/3秒間隔で逆算した近似値を使って判定している（詳細は analyze_temporal_attention.py
     冒頭のdocstring「Ground Truthのフェーズ境界判定」参照）
- max_attention_frame, max_attention_value: 30フレーム中でAttentionが最大だったフレームと値
- plot_path, raw_attn_path: 対応するプロット・生データのパス
- frame_01_attention 〜 frame_30_attention: フレームごとのTemporal Attention重要度
  （query=最後のフレーム、head平均・layer平均・空間パッチ平均後の値。集約方法の詳細は
  analyze_temporal_attention.py 冒頭のdocstringを参照）

## Attentionの解釈について（重要）
注意: Attentionが高い = そのフレームが分類の直接的な原因、とは限らない。ここでは「モデルが時間方向のどこを平均的に参照する傾向があるか」を示す。
Attentionは中間表現同士の類似度に基づく重み付けであり、モデルの最終判断に対する
厳密な因果的説明ではない。ここでの結果は「モデルが時間方向のどこを平均的に参照する
傾向があるか」を示す記述的な指標として解釈すること。

## 研究上の見方の例
- has_transition=Trueのウィンドウで、max_attention_frameがtransition_frame付近に
  集中しているか（＝切り替わり周辺の情報を参照する傾向があるか）を、
  正解ウィンドウと誤分類ウィンドウで比較する。
- correct=Falseのウィンドウで、Attentionのピークが「間違えた予測先のフェーズが
  支配的な時間帯」に寄っていないか（＝古いフェーズの情報に引きずられて
  誤分類している可能性）を確認する。
- raw/*.npy を使えば、層(layer)ごとにAttentionパターンが変化するか
  （浅い層は自己参照が強く、深い層ほど遠いフレームを参照する、等の傾向）も分析できる。
