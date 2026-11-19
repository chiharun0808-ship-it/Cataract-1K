# Artificial intelligence across the cataract care continuum: a systematic scoping review introducing the CAICC framework and the surgical complexity prediction gap

- 引用キー：2026_tjokrovonco_artificial
- 著者：**医者**（6人全員が医学系。筆頭・責任著者 Tjokrovonco は Universitas Padjadjaran 医学部眼科 ／ Cicendo Hospital National Eye Center、インドネシア）
- 発表：2026-10（BMC Ophthalmology 26:696、2026-10-27 公開。受理 2026-10-22）
- リンク：https://doi.org/10.1186/s12886-026-05483-4 ／ PMC：https://pmc.ncbi.nlm.nih.gov/articles/PMC13718037
- 状態：流し読み（2026-11-20、PMC と PDF 本文を Claude が確認。数字は PDF で照合済み）
- 本人：読んだ（2026-11-20、和訳で）
- 全文の和訳：`2026_tjokrovonco_artificial_ja.pdf`（2026-11-20、手元のみ。GitHub には置いていない）

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
- 関係するテーマ：**他施設で使えるか**、**医者の視点**
- 医者側から「性能が高くても、他の施設で使えるか・臨床で役立つかとは別」と言っている → 研究の背景（汎化が必要な理由）の根拠に使える。
- 工程認識はこのレビューで独立した分類ではなく、Intelligent Surgery の中の一部。
- 医者が挙げる今後の優先事項：確立した応用の前向き検証、機器や患者集団が変わっても再現できること、AI が臨床の判断や結果を変えるかの評価。

## 引用したい文（原文どおり）
- "High technical performance should therefore not be equated with generalizability, clinical utility"
- "Most included studies remained at internal validation stages, and prospective multicenter evaluation was rare."
- "Such models should define clinically actionable outcomes, undergo independent external validation"
- 優先事項："... reproducibility across equipment and populations, ... and evaluation of whether AI changes clinical decisions or outcomes."

## 手術中の AI（Intelligent Surgery）の詳細 — PDF で確認済み（Table 1・Table 2・本文 p.8 付近）
- 28 本：工程（phase / step）認識、器具の検出・追跡、解剖の領域分割、術中ガイド、技能評価、手術時間の予測など（文献 144〜170）。
- 検証が当てはまる 27 本のうち（Table 1 Panel A、重複あり）：内部 24（88.9%）、時間的 0、外部 7（25.9%）、前向き 1（3.7%）、前向き多施設 0。
- 到達した最高段階（Panel B）：なし・不明 1、**内部 19（70.4%）**、外部 6（22.2%）、前向き単施設 1（3.7%）、**多施設 0**。
- **工程認識が一番多いタスク**。正解率 82.64〜98.79%（6 本）、F1 77.67〜98.79%（3 本）、precision 82.20〜94.90%（2 本）、recall 77.32〜95.20%（2 本）、AUC 0.950〜0.990（2 本）。プールした値ではなく、報告値の幅。
- その他：瞳孔の領域分割 Dice 96.82%（1 本）、技能評価 AUC 0.780〜0.880（3 本）、器具の識別 AUC 0.970〜0.999（2 本）。
- 著者の評価：ほとんどが過去の手術動画と内部評価に頼っている。日常の臨床で使う前に、**固定したモデルを、別の術者・施設・顕微鏡・症例の難しさで前向きに試す必要がある**。
- 今後の課題（Table 2）："Real-time multicenter evaluation across surgeons, equipment, workflows, and case complexity"

### 含まれている白内障の工程認識・手術動画の論文（読む候補）
- 167 Yeh ら「PhacoTrainer: A Multicenter Study of Deep Learning for Activity Recognition in Cataract Surgical Videos」（TVST）← **多施設**
- 168 Morita, Tabuchi ら「Real-Time Surgical Problem Detection and Instrument Tracking in Cataract Surgery」（J Clin Med 2020）← **リアルタイム、日本の眼科**
- 169 Wang, Nakashima ら「Real-time estimation of the remaining surgery duration for cataract surgery …」← **リアルタイム、工程時間**
- 160 Fang ら「Global–local multistage TCN for cataract surgery phase recognition」（Biomed Eng Online 2022）
- 153 Mahmoud ら「CatStep: … Phase Classification and Boundary Segmentation …」
- 151 Nasirihaghighi, Ghamsarian ら「Dual invariance self-training for reliable semi-supervised surgical phase recognition」（IEEE ISBI 2025、Cataract-1K の著者）
- 161 Touma ら「code-free machine learning model for the classification of cataract surgery phases」（Sci Rep 2022、眼科医）

## 確認しておいたこと
- 「手術の難しさは、1 つの合併症が起こる確率と同じではない」という文は本文にある（原文："surgical complexity is not synonymous with the probability of a single complication"）。
- 予測の揺れ（安定性）や正解ラベルの質について直接書いた文は、PDF 本文にもない。
- 外部検証で見るべき分布のずれ：施設、患者集団、機器、術者、ワークフロー。前向き研究では性能だけでなく、較正、失敗のしかた、使いやすさ、臨床判断への影響も評価すべき、とある。「手術中の判断に影響するシステムでは特に多施設評価が重要」。
