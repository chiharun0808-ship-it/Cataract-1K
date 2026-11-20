# DINOv2: Learning Robust Visual Features without Supervision

- 引用キー：2024_oquab_dinov2
- 著者：**工学系**。Maxime Oquab ほか（Meta AI Research、一部 Inria）。
- 発表：2024-01（Transactions on Machine Learning Research）／arXiv 2023-04（2304.07193、PDF は v2）
- リンク：https://arxiv.org/abs/2304.07193 ／ https://openreview.net/forum?id=a68SUt6zFt ／ コード・重み https://github.com/facebookresearch/dinov2（Apache 2.0）
- ライセンス：arXiv 標準（翻訳不可）→ 全文和訳ではなく**日本語の詳しい要約** `2024_oquab_dinov2_ja.pdf`（2026-11-20、手元のみ。本文 p.1–21。付録は未対応）
- 状態：未読（日本語要約あり）

## 一言で
整理した 1.42 億枚（LVD-142M）で、DINO＋iBOT の自己教師あり学習を 1B の ViT-g まで大きくし、小さいモデルへ蒸留。**凍結したまま**線形層だけで、多くのタスクで CLIP 系と互角以上。

## 中身の要点（数字は原文で照合済み）
- 損失：画像レベルの DINO、パッチレベルの iBOT（マスクして予測）、ヘッドは別々、Sinkhorn-Knopp、KoLeo（特徴を広げる）、最後だけ 518 の高解像度。
- ImageNet 線形：ViT-g/14 86.5%（OpenCLIP ViT-G 86.2%）。微調整しても +2% 程度 → 凍結で十分。
- **動画**（凍結＋線形、8 フレームを等間隔）：K400 78.4、UCF101 91.2、SSv2 38.3（ViT-g）。UCF・K400 は特徴を平均、SSv2 は連結。ViT-L は 76.3 / 90.5 / 35.6。
- 密なタスク：ADE20k 線形 49.0（ViT-g）、パッチ特徴の PCA で前景と背景・物体の部位が分かれる。
- アブレーション：KoLeo なしだと検索が 8% 以上悪化、iBOT の MIM なしだとセグメンテーションが約 3% 悪化。蒸留した ViT-L はゼロからの ViT-L より全 12 項目で上。

## 自分の研究との関係
- 関係するテーマ：手法一般（特徴抽出器の出典）
- 「長い窓」の枝の特徴に使う DINOv2 の出典。凍結したまま使う根拠（微調整の伸びが小さい）。
- 動画を「フレームごとの特徴の平均」で扱うと SSv2 のような動きが大事なデータでは弱い、という点は、フレーム特徴を時間方向にどう束ねるかを考えるときの参考。

## 読むときに確かめたいこと
- 自分が使っている大きさ（S/B/L/g）とレジスタ付きかどうか、入力解像度をメモする。
