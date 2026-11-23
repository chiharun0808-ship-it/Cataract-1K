# Global–local multi-stage temporal convolutional network for cataract surgery phase recognition

- 引用キー：2022_fang_global
- 著者：Fang L, Mou L, Gu Y, Hu Y, Chen B, Chen X, Wang Y, Liu J, Zhao Y（中国科学院 慈渓生物医学工程研究所ほか工学系と、愛爾眼科医院 上海・済南）。**混合**
- 発表：2022-11（BioMedical Engineering OnLine 21:82、2022-11-30 公開）
- リンク：https://doi.org/10.1186/s12938-022-01048-w ／ PMC9710114
- ライセンス：CC BY 4.0 → 全文和訳できる
- 状態：未読（2026-11-24 登録）

## 一言で
白内障の工程認識を、**瞳孔の領域・器具の領域・フレーム全体の 3 つの流れ**で特徴を取り、多段の時間畳み込み（MS-TCN）でつなぐ GL-MSTCN。

## 結果（要旨より）
- 自前の CSVideo（32 本、1 施設）で 95.8%、公開の Cataract-101 で 96.5%（正解率）。
- 本文に「1 つの病院のデータでデータの多様性が少ない」という限界の記載あり。

## 自分の研究との関係
- 器具と瞳孔の周りを別に見る設計は、器具の先端の周りを見る案（tip-attention）の先行例。
- Cataract-101 での結果は比較に使える。[[2026_tjokrovonco_artificial]] の手術中の AI の文献 160 として含まれている。
