# Artificial intelligence across the cataract care continuum: a systematic scoping review introducing the CAICC framework and the surgical complexity prediction gap

- 引用キー：2026_tjokrovonco_artificial
- 著者：**医者**（6人全員が医学系。筆頭・責任著者 Tjokrovonco は Universitas Padjadjaran 医学部眼科 ／ Cicendo Hospital National Eye Center、インドネシア）
- 発表：2026-10（BMC Ophthalmology 26:696、2026-10-27 公開。受理 2026-10-22）
- リンク：https://doi.org/10.1186/s12886-026-05483-4 ／ PMC：https://pmc.ncbi.nlm.nih.gov/articles/PMC13718037
- 状態：流し読み（2026-11-20、PMC の本文を Claude が確認。「Intelligent Surgery」の結果の部分は要約でしか読めていない → 下の「要確認」）

## 一言で
白内障の診療の流れ全体（見つける→評価→手術計画→手術中→回復）の AI 研究 185 本を整理した、眼科医によるレビュー。ほとんどが内部での検証どまりで、複数施設での前向き評価はごくわずか。

## 何をしたか
- PRISMA-ScR に沿ったスコーピングレビュー。PubMed・Scopus・Cochrane（＋灰色文献）。2020-01〜2025-07-20。185 本を採用。
- **IEEE Xplore は検索していない**（著者自身が限界として書いている）→ 工学系の工程認識の論文は多くが漏れているはず。
- CAICC（Cataract AI Clinical Continuum）という枠組みで分類：Detection 52 ／ Characterization 50 ／ Personalized Surgical Planning 27 ／ **Intelligent Surgery 28**（手術動画、工程認識、器具、技能評価など）／ Recovery 24 ／ Patient Engagement 4。
- 検証の成熟度（到達した最高段階）：なし・不明 → 内部 → 時間的 → 外部 → 前向き単施設 → 前向き多施設。

## 結果（数字）
- 検証の対象になる 181 本：内部 123（68.0%）、外部 33（18.2%）、前向き単施設 6（3.3%）、前向き多施設 2（1.1%）、なし・不明 17（9.4%）。
- 「手術の難しさ」を術前に総合的に予測する、臨床で検証されたモデルは見つからなかった（= 題名の surgical complexity prediction gap）。症例の割り振り、手術の準備、個別のリスク管理に必要だとする。

## 自分の研究との関係
- 問い：**Q4（他の施設でも使えるか）**、**Q2（医者は何を求めているか）**
- 医者側から「性能が高くても、他の施設で使えるか・臨床で役立つかとは別」と言っている → 研究の背景（汎化が必要な理由）の根拠に使える。
- 工程認識はこのレビューで独立した分類ではなく、Intelligent Surgery の中の一部。
- 医者が挙げる今後の優先事項：確立した応用の前向き検証、機器や患者集団が変わっても再現できること、AI が臨床の判断や結果を変えるかの評価。

## 引用したい文（原文どおり）
- "High technical performance should therefore not be equated with generalizability, clinical utility"
- "Most included studies remained at internal validation stages, and prospective multicenter evaluation was rare."
- "Such models should define clinically actionable outcomes, undergo independent external validation"
- 優先事項："... reproducibility across equipment and populations, ... and evaluation of whether AI changes clinical decisions or outcomes."

## 要確認（PDF で自分で読む）
- Intelligent Surgery の結果（要約でしか読めていない）：28 本のうち、最高段階は内部 19・外部 6・前向き単施設 1、多施設なし。工程認識の正解率 82.64〜98.79%（6 件）、F1 77.67〜98.79%（3 件）。リアルタイムで使った研究はなく、「リアルタイムの多施設評価」が今後の課題、とあるらしい。→ 引用する前に本文で数字を確かめる。
- 予測の揺れ（安定性）やラベルの質について直接書いた文は見つかっていない。
