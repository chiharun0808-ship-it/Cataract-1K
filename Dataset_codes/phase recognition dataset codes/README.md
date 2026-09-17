# phase recognition dataset codes

手術フェーズ認識の学習・評価・推論・データ前処理・Attention解析一式。

**実行は必ずこのディレクトリ直下をカレントディレクトリにして行う**こと
（例: `python training/train2.py`、`python evaluation/test2.py <checkpoint_dir>`）。
CSVパス・チェックポイント出力先・動画パスなどはすべてこのディレクトリからの
相対パスとして書かれている。

## ディレクトリ構成

- `models/`, `dataset_index/`, `dataset_weight/`, `utils/` — 学習・評価・推論・解析
  スクリプトから共通で import される「ライブラリ」。単体では実行しない。
- `training/` — 学習スクリプト（train2.py: クリップ単位, train3.py: スライディング
  ウィンドウ方式）
- `evaluation/` — test split 評価スクリプト（test2.py）
- `inference/` — 学習済みモデルを動画に適用するスクリプト・シェルスクリプト
- `data_preparation/` — 動画からの学習用データ生成（build_sliding_window_index.py,
  dataset_preprocessing/ 以下の抽出・変換スクリプト群）
- `attention_analysis/` — TimeSformer の Attention 解析一式
  - `scripts/` — 解析スクリプト本体
  - `temporal/`, `spatial_correct/`, `spatial_errors/`, `frame11_bias_check/` —
    各解析の出力（README・csv・plots・raw）
  - `logs/` — 各解析実行時のログ
- `debug/check/` — 簡易動作確認用スクリプト
- `archive/` — 現在未使用・正式版に置き換え済みの参考スクリプトとその入出力
  （train.py, video_spllit.py 等。動作しないものを含む）
- `data/` — 学習・評価で使うインデックスCSV（dataset_index*.csv）。
  dataset_index/create_index.py・data_preparation/build_sliding_window_index.py が生成する
- `logs/` — 学習・前処理などの実行ログ
- `outputs/` — スクリプトから参照されない単発の集計結果（cm_sum_5fold.npy,
  phase_duration*.csv）
- `sliding_window_dataset/`, `sliding_window_dataset_stride5/`, `result/` — 生成された
  クリップ・チェックポイント・推論結果（サイズが大きいため git 管理対象外）
