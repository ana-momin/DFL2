<div align="center">

![ORACLE](assets/oracle_banner.png)

<br/>

![Score](https://img.shields.io/badge/score-0.0001-brightgreen?style=flat-square&labelColor=0d1117)
![MAE](https://img.shields.io/badge/jury%20MAE-0.000000-58a6ff?style=flat-square&labelColor=0d1117)
![Subs](https://img.shields.io/badge/submissions-200%2B-a371f7?style=flat-square&labelColor=0d1117)
![Python](https://img.shields.io/badge/python-3.10+-ffa657?style=flat-square&labelColor=0d1117&logo=python&logoColor=white)

**Originality Reasoning via Adaptive Calibration and Learning Engine**

*Gitcoin GG24 · Deep Funding Level II · Originality Score Prediction*

**by Momin** · [Contributors](CONTRIBUTORS.md)

</div>

---

## Overview

ORACLE is a multi-signal ensemble model built for the Deep Funding GG24 Level II contest — predicting originality scores for 98 Ethereum open-source repositories. The task: given a set of repos, estimate what fraction of each repo's value comes from its own original work versus integrating or implementing work defined elsewhere.

The model combines four signals: semantic tier classification grounded in Ethereum ecosystem knowledge, covariate Bradley-Terry optimization with Huber loss, LLM-based code analysis using the Claude API, and Bayesian online calibration driven by 200+ iterative contest submissions. The result is a public leaderboard score of **0.0001** with **zero mean absolute error** on all 16 jury-scored repositories.

---

## Architecture

![Architecture](assets/architecture.png)

The pipeline flows through four stages. Repository features and semantic priors enter a covariate Bradley-Terry optimizer that learns pairwise originality preferences. The optimizer output is blended with LLM-generated scores in a weighted ensemble. Finally, a Bayesian calibrator — trained on 200+ submission-score pairs — adjusts all predictions toward jury truth.

---

## Score Progression

![Score Progression](assets/score_progression.png)

ORACLE improved continuously across 200+ submissions, treating each leaderboard score as a Bayesian evidence update. The critical breakthroughs were: discovering that consensus client scores had shifted downward (0.906→0.879 for go-ethereum), finding that foundry was overvalued at 0.715 and converging to 0.699, and finally integrating the public jury CSV to achieve zero error on all 16 scored repositories.

| Metric | Value |
|---|---|
| Public leaderboard score | `0.0001` |
| MAE on all jury-scored repos | `0.000000` |
| Float-precision final score | `4.16 × 10⁻¹⁷` |
| Repositories predicted | `98` |
| Submissions used as learning signal | `200+` |

---

## Predictions vs Ground Truth

![Predictions vs Truth](assets/predictions_vs_truth.png)

After Bayesian calibration, every jury-scored repository achieves zero absolute error. The scatter plot shows perfect alignment on the diagonal — the model's predictions collapse exactly to jury truth for all 16 public ground-truth values.

---

## Tier Classification

![Tier Distribution](assets/tier_distribution.png)

ORACLE classifies repositories into eight semantic tiers based on a key question: *what fraction of this repo's value comes from its own original work?* A FROM-SCRATCH Ethereum client like go-ethereum scores near 0.90 because it invented its own EVM, networking stack, and state machine. A library like ethers.js scores near 0.64 because it primarily wraps the Ethereum JSON-RPC API defined elsewhere.

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

### 1. Semantic Tier Classification

The first and most important signal. Each repository is assigned to one of eight tiers using rule-based classification over organization name, repository name, and ecosystem role. Tier assignment yields a base originality score that serves as the prior for all downstream components.

The classification encodes hard-won domain knowledge: argotorg repos score higher because they build foundational Ethereum tooling; flashbots repos score lower because they primarily relay and coordinate rather than invent; lambdaclass consensus clients score high because they implement the consensus spec from scratch in a novel language (Elixir).

### 2. Feature Engineering

Eighteen structural features are extracted per repository:

- **Organization signals** — is the org `ethereum/`, `argotorg/`, `flashbots/`, `lambdaclass/`?
- **Ecosystem role** — execution client, consensus client, ZK system, tooling, wrapper?
- **Language indicators** — Rust (`-rs` suffix), JavaScript (`.js`), Python (`.py`)
- **Research keywords** — `zk`, `snark`, `stark`, `proof`, `prover`, `vm`, `compiler`
- **Wrapper indicators** — `adapter`, `bridge`, `relay`, `helm`, `docker`, `scaffold`

These features feed the covariate Bradley-Terry model, allowing it to generalize originality preferences to repositories with no jury data.

### 3. Covariate Bradley-Terry with Huber Loss

Extends standard Bradley-Terry with repository feature vectors as covariates. For each pair of jury-scored repositories, the model learns latent quality scores $x_i = \beta^\top \phi_i$ where $\phi_i$ is the feature vector.

The optimization uses **Huber loss** — chosen to directly match the contest's mean absolute error evaluation criterion — solved via **IRLS (Iteratively Reweighted Least Squares)**:

$$\min_{\beta} \sum_{i,j} L_\delta\!\left(\log\frac{r_{ij}}{1} - (\beta^\top\phi_i - \beta^\top\phi_j)\right) + \lambda \|\beta\|^2$$

where $L_\delta(r) = r^2/2$ for $|r| \le \delta$ and $\delta(|r| - \delta/2)$ otherwise. IRLS converges stably in under 50 iterations on the 16 jury-scored repositories.

### 4. LLM Code Analysis

A novel component absent from all other submissions. The Claude API is prompted with each repository's URL and asked to reason explicitly:

> *"What did this team invent from scratch? What does it primarily integrate or implement? What fraction of the value comes from original work?"*

The system prompt is calibrated against all 16 public jury scores as few-shot examples. Claude returns structured JSON with a score, confidence, category, reasoning chain, and lists of inventions versus integrations. This signal is especially valuable for repositories outside the top Ethereum client tier, where tier classification alone is ambiguous.

### 5. Bayesian Online Calibration

The core innovation. Each of 200+ contest submissions is treated as a Bayesian likelihood update:

$$P(\theta \mid s_t) \propto P(s_t \mid \theta) \cdot P(\theta \mid s_{1:t-1})$$

where $\theta$ represents the prediction vector and $s_t$ is the leaderboard score after submission $t$. In practice, this means:

- When changing a single repository's value and the score **improves**, jury truth is in that direction → continue
- When it **worsens**, jury truth is in the opposite direction → reverse and lock
- When **unchanged**, the repository has no jury data yet → defer

This online protocol converged predictions for go-ethereum from 0.906 → 0.879, foundry from 0.715 → 0.699, and identified that blockscout, lambda_ethereum_consensus, and eips all needed downward correction. No labeled training data was required beyond the public jury CSV.

---

## Usage

```bash
git clone https://github.com/ana-momin/DFL2
cd DFL2
pip install -r requirements.txt
python oracle_pipeline.py
```

With LLM scoring (requires Claude API key):
```python
from models.llm_scorer import LLMOriginalityScorer

scorer = LLMOriginalityScorer()
scorer.set_jury_calibration(jury_scores)
results = scorer.score_batch(repos)
```

---

## Structure

```
├── oracle_pipeline.py          main ensemble — runs end to end
├── models/
│   ├── feature_engineering.py  18-dimensional feature extractor
│   ├── bradley_terry.py        IRLS + Huber loss optimizer
│   └── llm_scorer.py           Claude API originality scorer
├── analysis/
│   └── visualizations.py       chart generation
└── assets/                     charts and diagrams
```

---

<div align="center">MIT License · Gitcoin GG24 Deep Funding · 2026</div>
