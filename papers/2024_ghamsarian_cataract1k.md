# Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos

- 引用キー：2024_ghamsarian_cataract1k
- 著者：混合（筆頭 Ghamsarian は工学系〔Bern 大〕。眼科医〔Klinikum Klagenfurt・Bern 大病院〕が共著）← 所属は PDF で確認する
- 発表：2024-04（Scientific Data 11:373）／ arXiv 2023-12（2312.06295）
- リンク：https://www.nature.com/articles/s41597-024-03193-4 ／ PMC：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11014927/
- 状態：未読（PDF 未取得。オープンアクセスなので取得できるはず）

## 一言で
自分が使っているデータセットの元論文。Klinikum Klagenfurt の白内障手術動画 1,000 本（2021〜2023）。工程認識用に 56 本へ 13 工程のフレーム単位ラベル（1024×768、30 fps）。領域分割と異常検出のサブセットもある。

## 読むときに確かめたいこと
- 13 工程の定義と、ラベルの付け方（誰が・どの基準で・境界の決め方）→ 医師の「正解ラベルが大まかなだけかも」という指摘（11-18）の確認に使う。
- 56 本の選び方、執刀医の数。
- 公式のベースラインの結果と分割。
