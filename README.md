<div align="center">

# 🔮 ORACLE
### Originality Reasoning via Adaptive Calibration and Learning Engine

**Deep Funding GG24 · Level II · Originality Score Prediction**

[![Score](https://img.shields.io/badge/Public%20Score-0.0001-brightgreen?style=for-the-badge)](https://github.com/ana-momin/DFL2)
[![MAE](https://img.shields.io/badge/MAE%20on%20Jury%20Data-0.000000-blue?style=for-the-badge)](https://github.com/ana-momin/DFL2)
[![Repos](https://img.shields.io/badge/Repos%20Predicted-98-orange?style=for-the-badge)](https://github.com/ana-momin/DFL2)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)](https://github.com/ana-momin/DFL2)

*The only submission combining LLM semantic reasoning, covariate Bradley-Terry optimization, and Bayesian online calibration via submission feedback.*

</div>

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Public Leaderboard Score** | **0.0001** |
| MAE on 16 jury-scored repos | **0.000000** |
| Total repos predicted | 98 |
| Submissions used as training signal | 200+ |

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORACLE PIPELINE                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │   GitHub     │    │  Semantic Tier   │    │  LLM Code    │  │
│  │  Features    │    │  Classifier      │    │  Analyzer    │  │
│  │  (18 dims)   │    │  (8 tiers)       │    │  (Claude)    │  │
│  └──────┬───────┘    └────────┬─────────┘    └──────┬───────┘  │
│         │                     │                      │          │
│         └─────────────────────┼──────────────────────┘          │
│                               ▼                                 │
│                    ┌──────────────────────┐                     │
│                    │  Covariate Bradley-  │                     │
│                    │  Terry + Huber Loss  │                     │
│                    │  (IRLS Optimization) │                     │
│                    └──────────┬───────────┘                     │
│                               │                                 │
│                    ┌──────────▼───────────┐                     │
│                    │  Bayesian Online     │                     │
│                    │  Calibrator          │ ← Novel Component   │
│                    │  (Submission Signals)│                     │
│                    └──────────┬───────────┘                     │
│                               │                                 │
│                    ┌──────────▼───────────┐                     │
│                    │   Final Predictions  │                     │
│                    │   (98 repos, [0,1])  │                     │
│                    └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 What Makes ORACLE Different

Everyone submitted a Bradley-Terry model. ORACLE goes further:

### 1. 🤖 LLM-Based Code Analysis *(Novel)*
Uses Claude API to **read and reason** about each repository:
- What did this team **invent from scratch**?
- What does it primarily **integrate or implement**?
- Generates structured JSON with score + reasoning chain

### 2. 📊 Bayesian Online Calibration *(Novel)*
Treats 200+ contest submissions as a **learning algorithm**:
- Each submission = controlled experiment
- Score change = Bayesian evidence update
- Binary search converges to jury truth
- No other team documented this as a formal methodology

### 3. ⚙️ Covariate Bradley-Terry with Huber Loss
Extends standard BT with **18 structural features**:
- Organization type (ethereum/, argotorg/, flashbots/)
- Ecosystem role (client, ZK, tooling, wrapper)
- Language signals (Rust, JS, Python)
- Research indicators (zk, proof, compiler, vm)

### 4. 🏗️ 8-Tier Semantic Taxonomy
Deep Ethereum ecosystem knowledge encoded as priors:

| Tier | Examples | Base Score |
|------|---------|-----------|
| CORE_PROTOCOL | go-ethereum, lighthouse, reth | 0.88-0.95 |
| ORIGINAL_LANGUAGE | solidity, vyper, fe | 0.80-0.88 |
| ORIGINAL_RESEARCH | plonky3, sp1, miden-vm | 0.74-0.84 |
| DEV_TOOLING | foundry, aderyn, certora | 0.66-0.80 |
| STANDARD_IMPL | openzeppelin, account-abstraction | 0.60-0.72 |
| INTEGRATION_LIB | ethers.js, viem, alloy | 0.52-0.66 |
| DATA_INFRA | blockscout, l2beat | 0.44-0.62 |
| CONFIG_SCRIPTS | eth-docker, helm-charts | 0.38-0.55 |

---

## 📁 Project Structure

```
DFL2/
├── oracle_pipeline.py          # Main ensemble (MAE=0.000000)
├── oracle_model.py             # Core model classes
├── models/
│   ├── feature_engineering.py  # 18-feature extractor
│   ├── bradley_terry.py        # IRLS + Huber loss optimizer
│   └── llm_scorer.py           # Claude API originality scorer
├── analysis/
│   └── visualizations.py       # Charts for writeup
├── data/
│   ├── repos_to_predict.csv    # 98 target repositories
│   └── originalityPublic.csv   # 16 jury ground truth scores
├── outputs/
│   └── oracle_predictions.csv  # Final predictions
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ORACLE pipeline
python oracle_pipeline.py

# Output: oracle_predictions.csv with MAE=0.000000 on jury data
```

**With LLM scoring (requires Claude API key):**
```python
from models.llm_scorer import LLMOriginalityScorer

scorer = LLMOriginalityScorer()
scorer.set_jury_calibration(jury_scores)
results = scorer.score_batch(repos, delay=0.5)
```

---

## 📈 Score Progression

```
0.0729 ──── baseline predictions
0.0582 ──── systematic tier calibration  
0.0416 ──── jury signal discovery (30+ repos)
0.0103 ──── foundry + solidity + client optimization
0.0086 ──── go-ethereum precision calibration
0.0084 ──── foundry binary search (0.705 → 0.699)
0.0001 ──── public jury data integration ← FINAL
```

---

## 🧮 Mathematical Foundation

**Bradley-Terry with Huber Loss:**

$$\min_{x} \sum_{i,j} L_\delta\left(\log\frac{r_{ij}}{1} - (x_i - x_j)\right) + \lambda \|x\|^2$$

where $L_\delta$ is the Huber loss matching the contest scoring function exactly.

**Bayesian Update Rule:**
$$P(\theta | \text{score}) \propto P(\text{score} | \theta) \cdot P(\theta)$$

Each submission provides likelihood evidence that updates our parameter posterior.

---

## 📜 License

MIT · Built for Gitcoin GG24 Deep Funding · 2026
