# Real-time artificial intelligence evaluation of cataract surgery: A preliminary study on demonstration experiment

- 引用キー：2022_tabuchi_realtime
- 著者：Hitoshi Tabuchi, Shoji Morita, Masayuki Miki, Hodaka Deguchi, Naotake Kamiura。**混合**：筆頭 Tabuchi は眼科医（広島大・ツカザキ病院 眼科）、Morita は Glory 社・兵庫県立大、Kamiura は兵庫県立大 工学研究科。筆頭は医者
- 発表：2022-04（Taiwan Journal of Ophthalmology 12(2):147–154、2022-04-13 公開）
- リンク：https://doi.org/10.4103/tjo.tjo_5_22 ／ PMC9262019
- ライセンス：CC BY-NC-SA（Europe PMC の表示）→ 非営利なら全文和訳できる
- 状態：未取得（PDF なし）。PMC・Europe PMC・出版社とも自動取得を拒否。ブラウザで https://pmc.ncbi.nlm.nih.gov/articles/PMC9262019/ から保存して `2022_tabuchi_realtime.pdf` にする

## 一言で
手術中に AI が危険度をリアルタイムで出すシステムを実際の手術で動かし、研修医と指導医を比べた予備研究。厚労科研 [[2023_oshika_aieye]] の「危険度」の論文版。

## 要旨から分かること（2026-11-22）
- 研修医 3 名（経験 100 例未満）と指導医 3 名（1,000 例以上）が各 3 例、計 18 例。
- 危険度（0 が最も安全、1 が最も危険）：CCC で研修医 0.556 対 指導医 0.433、phaco で 0.511 対 0.377。どちらも指導医が有意に良い（Wilcoxon、P = 0.0003、P < 0.0001）。
- 危険度は 0.1 秒ごとのサンプルを 5 つ移動平均して出す。

## 読むときに確かめたいこと
- 危険度の正解（何を「危険」としたか）と、工程の判定をどう組み合わせているか（[[2020_morita_realtime]] の D(t) と同じか）。
- 18 例・術者 6 名だけなので、術者の違いと症例の難しさを分けられているか。
