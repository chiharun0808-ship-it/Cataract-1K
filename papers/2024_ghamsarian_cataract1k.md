# Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos

- 引用キー：2024_ghamsarian_cataract1k
- 著者：混合。筆頭 Ghamsarian は Bern 大 CAIM（Department of Medicine の AI センター、工学系）。眼科（Klinikum Klagenfurt の Department of Ophthalmology、Bern の Inselspital の Department of Ophthalmology）と Klagenfurt 大の情報技術科が共著（PDF で確認）
- 発表：2024-04（Scientific Data 11:373）／ arXiv 2023-12（2312.06295）
- リンク：https://www.nature.com/articles/s41597-024-03193-4 ／ PMC：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11014927/
- ライセンス：CC BY 4.0（Scientific Data 版。arXiv 版は CC BY-NC-ND なので、使うのは雑誌版）→ 全文和訳できる
- 状態：未読（PDF は雑誌版を取得 2026-11-21、12 ページ）

## 一言で
自分が使っているデータセットの元論文。Klinikum Klagenfurt の白内障手術動画 1,000 本（2021〜2023）。工程認識用に 56 本へ 13 工程のフレーム単位ラベル（1024×768、30 fps）。領域分割と異常検出のサブセットもある。

## 読むときに確かめたいこと
- 13 工程の定義と、ラベルの付け方（誰が・どの基準で・境界の決め方）→ 医師の「正解ラベルが大まかなだけかも」という指摘（11-18）の確認に使う。
- 56 本の選び方、執刀医の数。
- 公式のベースラインの結果と分割。
