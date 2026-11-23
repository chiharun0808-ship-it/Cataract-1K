# CauCLIP: Bridging the Sim-to-Real Gap in Surgical Video Understanding via Causality-Inspired Vision-Language Modeling

- 引用キー：2026_he_cauclip
- 著者：He Y, Li A, Xue C（東南大学、南京）。**工学系**
- 発表：2026-02（arXiv 2602.06619 v1。5 ページの会議形式だが、採録先は未確認）
- リンク：https://arxiv.org/abs/2602.06619
- ライセンス：arXiv 標準（翻訳不可）→ 作るなら日本語の要約
- 状態：未読（2026-11-24 登録）

## 一言で
CLIP を使い、**学習に使っていない領域のデータなしで**（domain generalization）工程認識を頑健にする。周波数を使ったデータ拡張で領域に特有の性質（照明・質感・背景）を乱し、「因果でない」特徴を抑える損失を足す。

## 何をしたか
- 評価は MICCAI の SurgVisDom（VR の手術で学習 → 豚の実手術で試す、hard adaptation）。白内障ではない。
- 指標：weighted / unweighted / global F1、balanced accuracy。

## 自分の研究との関係
- 汎化の手法の例。凍結した基盤モデル（CLIP）を使う点は [[2026_ahmadi_cataractlmm]] の「凍結した DINO・CLIP が施設の違いに強い」と合う。
- 注意：ここでの「因果」は、画像の特徴のうち工程に効くもの・効かないものを分けるという意味。ユーザー案の「AをしたからB」という工程の順序の因果（[[causal-grammar-idea]]）とは別物。
