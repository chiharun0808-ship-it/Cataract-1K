# Multi-View Test-Time Adaptation for Semantic Segmentation in Clinical Cataract Surgery

- 引用キー：2025_li_multiview
- 著者：Li H, Ou M, Li H, Qiu Z, Niu K, Fu H, Liu J（南方科技大学 計算機・自律システム研究所、北京信息科技大学、A*STAR IHPC）。**工学系**
- 発表：2025-05（IEEE Trans Med Imaging 44(5):2307–2318）。オンラインの月は未確認
- リンク：https://doi.org/10.1109/TMI.2025.3529875 ／ PMID 40030926 ／ コード・データ https://github.com/liamheng/CAI-algorithms
- ライセンス：IEEE（オープンアクセスではない）
- 状態：未取得（PDF なし）（2026-11-24 登録）

## 一言で（要旨より。本文は未確認）
白内障手術の領域分割で、施設が変わると組織の見え方や照明が違って性能が落ちる問題に、**テスト時適応（使う施設のデータで、その場でモデルを更新）**で対処する MUTA を提案。データを 1 か所に集めずに済むのでプライバシーの問題がない。

## 何をしたか（要旨より）
- 学習時：複数の decoder（multi-view）で頑健にする。推論時：multi-view の知識蒸留でテスト時適応。
- 評価は「複数の白内障データセットで施設間を模した設定」（どのデータセットかは本文で確認）。

## 自分の研究との関係
- 白内障で施設間のずれを直接扱った工学系の論文。工程認識ではなく領域分割。
- 大鹿班報告書の「展開先でのファインチューニング」の課題に対し、ラベルなしでその場で合わせる方向の例。
- 入手：IEEE Xplore は有料。著者の GitHub にコードとデータがある。
