# Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos

- 引用キー：2024_ghamsarian_cataract1k
- 著者：混合。筆頭 Ghamsarian は Bern 大 CAIM（Department of Medicine の AI センター、工学系）。眼科（Klinikum Klagenfurt の Department of Ophthalmology、Bern の Inselspital の Department of Ophthalmology）と Klagenfurt 大の情報技術科が共著（PDF で確認）
- 発表：2024-04（Scientific Data 11:373）／ arXiv 2023-12（2312.06295）
- リンク：https://www.nature.com/articles/s41597-024-03193-4 ／ PMC：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11014927/
- ライセンス：CC BY 4.0（Scientific Data 版。arXiv 版は CC BY-NC-ND なので、使うのは雑誌版）→ 全文和訳できる
- 状態：未読（和訳あり、2026-11-21。PDF は雑誌版、12 ページ）

## 一言で
自分が使っているデータセットの元論文。Klinikum Klagenfurt の白内障手術動画 1,000 本（2021〜2023）。工程認識用に 56 本へ 13 工程のフレーム単位ラベル（1024×768、30 fps）。領域分割と異常検出のサブセットもある。

## 読むときに確かめたいこと
- 13 工程の定義と、ラベルの付け方（誰が・どの基準で・境界の決め方）→ 医師の「正解ラベルが大まかなだけかも」という指摘（11-18）の確認に使う。
- 56 本の選び方、執刀医の数。
- 公式のベースラインの結果と分割。

## 和訳を作るときに気づいたこと（2026-11-21）
- 工程の開始・終了の定義は書かれていない。CSV に各工程の最初と最後のフレームがあるだけ。アノテーションは筆頭（工学系）と眼科の D. Putzgruber-Adamitsch が一部を付け、残りを見直した（著者の貢献欄）。工程の専門的な情報は眼科医 2 名（Y.E.、D.P.）が提供。
- 待機（idle）は「器具を持ち替えていて器具が見えない時間」と定義。
- 公式の工程認識の結果は**一対他の 2 値分類**（3 秒クリップ、数をそろえたもの）で、13 クラスの正解率ではない。32 本で学習、残りでテスト。粘弾性物質の注入と前房洗浄は見た目が似ているので評価ではまとめた（最も成績が低い）。
- 表 6〜8 は PDF の中で文字が図形になっていて数字を取り出せないので、和訳では原文の表を画像で載せた。
