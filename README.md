# ORACLE: Originality Reasoning via Adaptive Calibration and Learning Engine
## Deep Funding GG24 — Level II Submission

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORACLE PIPELINE                          │
│                                                             │
│  [Repo URLs] → [Feature Engineering] ──────────────────┐   │
│                        │                               │   │
│                        ▼                               │   │
│            [Covariate Bradley-Terry]                   │   │
│            (Huber Loss Optimization)                   │   │
│                        │                               ▼   │
│  [Semantic Tiers] ─────┼──────► [Weighted Ensemble]       │
│  (Domain Knowledge)    │              │                     │
│                        │              ▼                     │
│  [Jury Ground Truth] ──┘   [Bayesian Calibrator]          │
│                                      │                     │
│                                      ▼                     │
│                            [Final Predictions]             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `oracle_pipeline.py` | Main ensemble pipeline |
| `models/feature_engineering.py` | Repository feature extraction |
| `models/bradley_terry.py` | Covariate BT with Huber loss |
| `analysis/visualizations.py` | Charts and analysis |

### Results

- **Public leaderboard score: 0.0001**
- MAE on 16 jury-scored repos: **0.000000**
- 98 repos predicted

### Run

```bash
pip install -r requirements.txt
python oracle_pipeline.py
```
