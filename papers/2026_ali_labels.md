# Labels for automated step recognition in cataract surgery for videos in the Cataract-1K dataset

- 引用キー：2026_ali_labels
- 著者：**混合**。筆頭 Ibrahim Dehshini Ali と最終著者 Shameema Sikder は Wilmer Eye Institute（Johns Hopkins 大の眼科）。Shah・Xia・Peven・Patel らは JHU の計算機科学、Vedula・Narayanan は Malone Center（医療工学）。著者 12 名。StepAL（2025_shah_stepal）と同じグループ。
- 発表：2026-10（Scientific Data、2026-10-21 オンライン公開。早期公開版〔accelerated article preview〕で、巻・論文番号は未定）
- リンク：https://www.nature.com/articles/s41597-026-08528-x ／ DOI 10.1038/s41597-026-08528-x
- データ：Johns Hopkins Research Data Repository（DOI 10.7281/T1ZZX1NJ と思われる。V2 は 2026-08-14。ページは自動取得で 403 のため未確認）
- 状態：未読（要旨だけ確認。PDF は自動取得できず、ブラウザでの取得待ち）

## 一言で（要旨より）
Cataract-1K の **1,000 本すべて**に手順（step）のラベルを付けたデータ論文。**386 本は手作業**で、**境界の基準を明示したプロトコル**で **10 手順**をラベル付け。アノテーター間の一致は非常に高い（ICC 0.994〜1.000、平均 IoU 0.980〜0.993）。その 386 本で最新の深層学習モデル 5 つを学習し、1,000 本すべてに疑似ラベルを付けた。手作業のラベル、疑似ラベル、予測確率を公開。

## 自分の研究との関係
- 問い：**Q1（揺れはなぜ起きるか）**、Q4（他の施設でも使えるか）
- 医師の指摘「揺れは正解ラベルが大まかなだけかも」（11-18）に直接関わる：境界の基準を明示し、一致度を測っている → 「ラベルの境界はどこまで決められるか」の根拠になる。
- 今使っている 56 本・13 工程（Cataract-1K の元のラベル）とは**別のラベル体系**（10 手順）。対応関係を確認する必要あり。
- 学習データが 56 本 → 386 本（手作業）に増やせる可能性。

## 読むときに確かめたいこと
- 10 手順の定義と境界の基準（元の 13 工程との対応）。
- アノテーターは誰か（眼科医か）、何人、どう一致度を測ったか。
- 5 つのモデルの名前と性能、データの分割。
- ライセンスとダウンロードの方法。
