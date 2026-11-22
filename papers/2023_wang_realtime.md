# Real-time estimation of the remaining surgery duration for cataract surgery using deep convolutional neural networks and long short-term memory

- 引用キー：2023_wang_realtime
- 著者：Bowen Wang, Liangzhi Li, Yuta Nakashima, Ryo Kawasaki, Hajime Nagahara。**混合**：Wang・Li・Nakashima・Nagahara は阪大 データビリティフロンティア機構（工学系）、Kawasaki は阪大 医学系の視覚情報学・眼科専門医（アノテーションも担当）。筆頭は工学系
- 発表：2023-05（BMC Medical Informatics and Decision Making 23:80、2023-05-04 公開）
- リンク：https://doi.org/10.1186/s12911-023-02160-0 ／ PMC10161556
- ライセンス：CC BY 4.0 → 全文和訳できる
- 状態：未読（2026-11-22 登録。厚労科研『AIの眼』の手術時間推定の論文版）

## 一言で
手術の残り時間を、CNN＋LSTM でリアルタイムに推定する。手作業のラベルは不要で、主な術者で学習したモデルを少しの動画で他の術者に移せる。

## 要旨から分かること（2026-11-22）
- 主な術者の手術時間は平均 318.7 秒（SD 83.4）。最良モデルの誤差（MAE）19.4 秒（平均時間の約 6.4%）。統計だけの素朴な方法より 35.5 秒小さい。
- **他の術者への移し替え**：事前学習の 20% の量で微調整すると、他の 2 名で MAE 28.3 秒・30.6 秒。術者ごとに一から学習したモデルより 8.1 秒・7.5 秒小さい。
- Cataract-101 での外部検証で TimeLSTM・RSDNet・CataNet を上回る。
- **厚労科研の報告書（[[2022_oshika_aieye]]・[[2023_oshika_aieye]]）では MAE 5.2〜9.3 秒**と書かれていて、この論文の数字（19.4 秒など）と合わない。条件の違いかどうか、本文で確かめる。

## 自分の研究との関係
- 術者が違うと性能が落ち、少しの微調整で戻る、という汎化の具体例（白内障手術・日本のデータ）。
