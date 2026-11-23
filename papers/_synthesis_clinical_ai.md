# 総括：眼科医から見た白内障 AI の現状と今後の研究課題、試せるデータセット

- 作成：2026-11-23（Claude がまとめ、ユーザーの依頼で保存）
- 元にした 5 本：[[2026_tjokrovonco_artificial]]・[[2026_zhang_advances]]・[[2026_tabuchi_grounded]]・[[2023_oshika_aieye]]・[[2024_tabuchi_largescale]]
- **【推測】** と書いた部分は論文に書かれていない Claude の解釈。それ以外は原文に書かれていること（原文の文とページを付けた）。
- データセットの入手状況は 2026-11-23 に確認したもの。

## 1. これまでどんな研究がされてきたか

| 論文 | 何をした研究か | 対象 |
|---|---|---|
| Tjokrovonco 2026（BMC Ophthalmol） | 白内障 AI の研究 185 本を、診療の流れ（見つける→評価→手術計画→手術中→回復）に沿って整理したスコーピングレビュー | 白内障の AI 全般 |
| Zhang 2026（Curr Opin Ophthalmol） | 白内障手術の進歩全体の総説。AI は一つの節だけ | 手術全般 |
| Tabuchi 2026 grounded（BMJ Health Care Inform） | 眼科 AI の設計の総説。3 層の設計を提案 | **網膜の画像と言語の AI が題材。手術動画ではない** |
| 大鹿班 R4 報告書 2023（厚労科研） | 手術動画の収集、危険度のリアルタイム表示、AI を他施設へ広げるときの障害の整理 | 白内障手術（動画と取り違え防止） |
| Tabuchi 2024（BMJ Qual Saf） | 取り違え防止 AI を 37,529 件の実臨床で使った結果 | **写真による認証。動画ではない** |

手術中の AI について分かっていること：

- 工程認識は「フレーム単位で 90% 超」まで来ている。
  > "Deep-learning systems trained on large, annotated cataract-surgery video datasets now identify key operative phases … with frame-level accuracies above 90%"
  > — Zhang 2026、AI の章「Surgical error detection and planning」の節（PMC の XML から読んだのでページ番号なし）
- ただし、ほとんどが過去の動画を使った、同じデータの中での評価。手術中の AI 28 本のうち、内部での評価どまりが 70.4%、多施設での前向き評価は 0 本。
  > "most studies have relied on retrospective surgical video datasets and internal evaluations."
  > — Tjokrovonco 2026、p.10

## 2. 今後求められる研究課題

### 課題 A：固定したモデルを、別の施設・術者・機器で評価する（5 本中 4 本が言及）
> "Prospective testing of locked models across different surgeons, institutions, microscope systems, and levels of case complexity remains necessary before routine clinical implementation."
> — Tjokrovonco 2026、p.10

> "External validation should evaluate clinically relevant distribution shifts, including different institutions, populations, devices, surgeons, and workflows."
> — Tjokrovonco 2026、p.14

> "larger multicentre studies … are necessary to confirm its generalisability"
> — Tabuchi 2024、p.9

> 「開発元と全く同様のソフトウェアや運用での展開は困難であり」
> — 大鹿班 R4 報告書、p.6

### 課題 B：リアルタイムでの評価
> "Real-time multicenter evaluation across surgeons, equipment, workflows, and case complexity"
> — Tjokrovonco 2026、Table 2（p.12）。手術中の AI の今後の課題として

【推測】リアルタイムで使うと、予測の揺れや遅れが問題になると考える。5 本とも、揺れには直接触れていない。

### 課題 C：手術の難しさと合併症の予測
> "we found no clinically validated multimodal model addressing the broader construct of overall operative complexity"
> — Tjokrovonco 2026、p.14

> "some success has been reported, although these models are not yet in clinical use"
> — Zhang 2026、同じ節。後嚢破損の予測について

### 課題 D：時間がたって新しい機器が入ってきたときに、AI を更新し続けること
> "The issue of unlearned IOLs in 7.3% of cases highlights a significant challenge in keeping AI systems relevant … underscores the importance of adaptive AI models and continuous updates"
> — Tabuchi 2024、p.8

【推測】この論文は写真による認証の話。手術動画に置き換えると、新しい器具や顕微鏡が入ったときに性能が落ちる問題に当たると考える。

### 課題 E：施設ごとの違いに合わせることを、仕組みとして持たせる
> 「今後、何らかのAI を横展開する際に、状況毎に新規開発を行うことは現実的ではない」
> — 大鹿班 R4 報告書、p.6

> "Layer 2 is explicit local policy: thresholds, report style, workflow routing and follow-up rules that differ legitimately between institutions"
> — Tabuchi 2026 grounded、p.3

【推測】grounded は網膜の AI が題材。手術動画では、施設による工程の順序やラベルの付け方の違いを、モデルの中に埋め込まずに別の層で調整することに当たると考える。

### 課題 F：正解率以外の評価（較正、失敗のしかた、実際に使われるか）
> "Prospective studies should assess not only technical discrimination but also calibration, failure modes, usability, clinical decision consequences, workflow integration"
> — Tjokrovonco 2026、p.14

> "errors in non-authenticated cases underscore the importance of …"
> — Tabuchi 2024、p.1 要旨。ミスは AI を使わなかった場面で起きた、という指摘

## 3. 試せる、ネットからダウンロードできるデータセット（2026-11-23 確認）

| データセット | 施設 | 中身 | 入手 | 使える課題 |
|---|---|---|---|---|
| **Cataract-1K**（使用中） | オーストリア（Klinikum Klagenfurt）、2021〜2023 | 工程 13 種・56 本、領域分割、異常のある症例 | Synapse で公開 | A・C（異常のある症例） |
| **Cataract-101** | オーストリア（ITEC Klagenfurt）、2018 公開 | 101 本、工程 10 種、術者の経験（低・高）と術者 ID | ITEC の FTP / Zenodo、CC BY-NC | A（術者の違い）、技能 |
| **CATARACTS** | フランス（Brest） | 50 本、器具が映っているか。工程のラベルは別の論文（DeepPhase）が作成 | IEEE DataPort（無料登録が要る） | A（別の国・施設） |
| **SICS-155** | インド（Sankara） | 155 本、工程 18 種。術式が違う（手動の小切開法） | Zenodo、CC BY-NC、そのまま取れる | A（施設と術式の違い） |
| **CataComp-104** | インド（Sankara） | 104 本、うち合併症 44 例（虹彩脱出・後嚢破損・硝子体脱出） | Hugging Face（条件に同意すると取れる） | C（合併症） |
| OphNet | YouTube（多くの施設） | 眼科手術 2,278 本、工程 102 種（白内障以外も含む） | プロジェクトページで公開と記載。条件は未確認 | A（ばらばらの施設） |
| ~~Cataract-LMM~~ | イラン 2 施設、3,000 本 | 工程・技能など | 2026-11-23 も Hugging Face の API で "disabled": true（ダウンロード不可） | （使えない） |

注意と【推測】：

- 【推測】施設の違いを試すなら、ラベルが揃っている Cataract-1K → CATARACTS / SICS-155 が一番手軽。ただしデータセットごとに工程の数と定義が違う（10・13・18 種など）ので、工程の対応表を作る必要がある。
- 【推測】Cataract-1K と Cataract-101 は同じクラーゲンフルトのデータで、公開の時期が離れている。施設の違いではなく、時間がたったときの変化（課題 D）を試す材料になるかもしれない。同じ病院で撮ったものかは原典で確かめていない。
- 課題 B（リアルタイム）は、どのデータでも動画を前から順に処理すれば試せる。ただし課題 F の「実際に使われるか」「臨床の判断が変わるか」は、公開データでは試せない（前向きの臨床研究が要る）。
- 【推測】顕微鏡の機種が違うときの評価（課題 A）に使える情報が各データにあるかは、確認していない。

## 出典（データセット）
- Cataract-101：https://ftp.itec.aau.at/datasets/ovid/cat-101 ／ https://zenodo.org/record/1220951
- CATARACTS：https://ieee-dataport.org/open-access/cataracts ／ 工程ラベル DeepPhase https://arxiv.org/pdf/1807.10565
- SICS-155：https://zenodo.org/records/19482928 ／ SICS-105：https://zenodo.org/records/13847781
- CataComp：https://arxiv.org/abs/2511.18968 ／ https://huggingface.co/datasets/SankaraEyeHospital/Catacomp-104
- OphNet：https://arxiv.org/abs/2406.07471 ／ https://minghu0830.github.io/OphNet-benchmark/
- Cataract-LMM：https://arxiv.org/abs/2510.16371
- Cataract-1K：https://pmc.ncbi.nlm.nih.gov/articles/PMC11014927
