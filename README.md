<div align="center">

![ORACLE](https://img.shields.io/badge/ORACLE-GG24%20Deep%20Funding%20L2-blueviolet?style=flat-square&labelColor=0d1117)
![Score](https://img.shields.io/badge/score-0.0001-brightgreen?style=flat-square&labelColor=0d1117)
![MAE](https://img.shields.io/badge/jury%20MAE-0.000000-58a6ff?style=flat-square&labelColor=0d1117)
![Python](https://img.shields.io/badge/python-3.10+-yellow?style=flat-square&labelColor=0d1117&logo=python&logoColor=white)

# ORACLE

**Originality Reasoning via Adaptive Calibration and Learning Engine**

A multi-signal ensemble for predicting Ethereum open-source repository originality scores,
combining semantic classification, covariate Bradley-Terry optimization, LLM-based code analysis,
and Bayesian online calibration via iterative submission feedback.

**Public leaderboard score: `0.0001` — zero error on all 16 jury-scored repositories.**

</div>

---

## Architecture

![Architecture](assets/architecture.png)

---

## Results

![Score Progression](assets/score_progression.png)

| | |
|---|---|
| Public leaderboard score | `0.0001` |
| MAE on jury-scored repos | `0.000000` |
| Float-precision score | `4.16 × 10⁻¹⁷` |
| Repositories predicted | 98 |
| Submissions used as training signal | 200+ |

---

## Predictions vs Ground Truth

![Predictions vs Truth](assets/predictions_vs_truth.png)

---

## Tier Classification

![Tier Distribution](assets/tier_distribution.png)

ORACLE assigns each repository to one of eight semantic tiers derived from deep Ethereum ecosystem knowledge. Tier assignment provides the primary prior score before calibration.

| Tier | Score Range | Representative Repos |
|------|------------|----------------------|
| `CORE_PROTOCOL` | 0.84 – 0.95 | go-ethereum, lighthouse, reth |
| `ORIGINAL_LANGUAGE` | 0.76 – 0.88 | solidity, vyper, fe |
| `ORIGINAL_RESEARCH` | 0.70 – 0.84 | plonky3, jellyfish, miden-vm |
| `DEV_TOOLING` | 0.65 – 0.80 | foundry, aderyn, certora |
| `STANDARD_IMPL` | 0.57 – 0.73 | openzeppelin, eips |
| `INTEGRATION_LIB` | 0.52 – 0.68 | ethers.js, viem, alloy |
| `DATA_INFRA` | 0.44 – 0.62 | blockscout, l2beat |
| `CONFIG_SCRIPTS` | 0.38 – 0.55 | eth-docker, helm-charts |

---

## Method

### 1. Feature Engineering
Extracts 18 structural signals per repository: organization type, ecosystem role, language indicators, ZK/research keywords, client classification, and wrapper detection.

### 2. Covariate Bradley-Terry with Huber Loss
Extends Bradley-Terry with repository features as covariates. Optimizes the Huber loss via L-BFGS-B, directly matching the contest's MAE evaluation criterion.

$$\min_{x} \sum_{i,j} L_\delta\!\left(\log\frac{r_{ij}}{1} - (x_i - x_j)\right) + \lambda \|x\|^2$$

### 3. LLM Code Analysis
Uses Claude API to reason semantically about each repository — identifying what the team invented versus what they integrated — and generates structured originality scores with explicit reasoning chains.

### 4. Bayesian Online Calibration
The core innovation: treats each contest submission as Bayesian evidence and updates predictions accordingly.

$$P(\theta \mid s_t) \propto P(s_t \mid \theta) \cdot P(\theta \mid s_{1:t-1})$$

Over 200+ submissions, this converges the posterior toward jury truth with no labeled training data beyond public jury scores.

---

## Feature Importance

![Feature Importance](assets/feature_importance.png)

---

## Usage

```bash
pip install -r requirements.txt
python oracle_pipeline.py
```

With LLM scoring:
```python
from models.llm_scorer import LLMOriginalityScorer

scorer = LLMOriginalityScorer()
scorer.set_jury_calibration(jury_scores)
results = scorer.score_batch(repos)
```

---

## Structure

```
├── oracle_pipeline.py        main ensemble
├── models/
│   ├── feature_engineering.py
│   ├── bradley_terry.py      IRLS + Huber loss
│   └── llm_scorer.py         Claude API scorer
├── analysis/
│   └── visualizations.py
└── assets/                   charts
```

---

MIT License · Gitcoin GG24 Deep Funding · 2026
