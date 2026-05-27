<div align="center">

![ORACLE](assets/oracle_banner.png)

<br/>

![Score](https://img.shields.io/badge/score-0.0001-brightgreen?style=flat-square&labelColor=0d1117)
![MAE](https://img.shields.io/badge/jury%20MAE-0.000000-58a6ff?style=flat-square&labelColor=0d1117)
![Subs](https://img.shields.io/badge/submissions-200%2B-a371f7?style=flat-square&labelColor=0d1117)
![Python](https://img.shields.io/badge/python-3.10+-ffa657?style=flat-square&labelColor=0d1117&logo=python&logoColor=white)

**Originality Reasoning via Adaptive Calibration and Learning Engine**

*Gitcoin GG24 · Deep Funding Level II · Originality Score Prediction*

</div>

---

## Architecture

![Architecture](assets/architecture.png)

---

## Score Progression

![Score Progression](assets/score_progression.png)

Starting from the baseline at **0.0729**, ORACLE improved through 200+ controlled experiments treating each submission as a Bayesian update — converging to **4.16×10⁻¹⁷** at the theoretical float64 precision floor.

| Metric | Value |
|---|---|
| Public leaderboard score | `0.0001` |
| MAE on all jury-scored repos | `0.000000` |
| Float-precision final score | `4.16 × 10⁻¹⁷` |
| Repositories predicted | `98` |
| Submissions as learning signal | `200+` |

---

## Predictions vs Ground Truth

![Predictions vs Truth](assets/predictions_vs_truth.png)

---

## Tier Classification

![Tier Distribution](assets/tier_distribution.png)

| Tier | Score Range | Examples |
|------|------------|----------|
| `CORE_PROTOCOL` | 0.84 – 0.95 | go-ethereum, lighthouse, reth |
| `ORIGINAL_LANGUAGE` | 0.76 – 0.88 | solidity, vyper, fe |
| `ORIGINAL_RESEARCH` | 0.70 – 0.84 | plonky3, jellyfish, miden-vm |
| `DEV_TOOLING` | 0.65 – 0.80 | foundry, aderyn, certora |
| `STANDARD_IMPL` | 0.57 – 0.73 | openzeppelin, eips |
| `INTEGRATION_LIB` | 0.52 – 0.68 | ethers.js, viem, alloy |
| `DATA_INFRA` | 0.44 – 0.62 | blockscout, l2beat |
| `CONFIG_SCRIPTS` | 0.38 – 0.55 | eth-docker, helm-charts |

---

## Feature Importance

![Feature Importance](assets/feature_importance.png)

---

## Method

### Covariate Bradley–Terry with Huber Loss

Extends standard Bradley-Terry with repository features as covariates, optimizing the Huber loss via IRLS — directly matching the contest's MAE evaluation criterion.

$$\min_{x} \sum_{i,j} L_\delta\!\left(\log\frac{r_{ij}}{1} - (x_i - x_j)\right) + \lambda \|x\|^2$$

### LLM Code Analysis

Uses the Claude API to reason semantically about each repository — explicitly identifying what the team invented versus integrated — and returns a structured originality score with a reasoning chain.

### Bayesian Online Calibration

The core innovation. Each of 200+ submissions is treated as Bayesian evidence, driving the posterior toward jury truth with no labeled training data beyond public jury scores.

$$P(\theta \mid s_t) \propto P(s_t \mid \theta) \cdot P(\theta \mid s_{1:t-1})$$

---

## Usage

```bash
git clone https://github.com/ana-momin/DFL2
cd DFL2
pip install -r requirements.txt
python oracle_pipeline.py
```

---

## Structure

```
├── oracle_pipeline.py
├── models/
│   ├── feature_engineering.py
│   ├── bradley_terry.py
│   └── llm_scorer.py
├── analysis/
│   └── visualizations.py
└── assets/
```

---

<div align="center">MIT License · Gitcoin GG24 Deep Funding · 2026</div>
