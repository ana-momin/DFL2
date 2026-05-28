"""
Cross-Validation & Evaluation — ORACLE 2.0
============================================
Rigorous model evaluation using leave-one-out cross-validation
on the public jury ground truth data.

Metrics computed:
  - MAE  (Mean Absolute Error)   ← contest scoring criterion
  - RMSE (Root Mean Squared Error)
  - R²   (coefficient of determination)
  - LOO-CV MAE (Leave-One-Out cross-validated MAE)
  - Per-tier breakdown
  - Residual analysis
"""

import numpy as np
import csv
from typing import Dict, List, Tuple, Optional


class ModelEvaluator:
    """
    Evaluates originality predictions against jury ground truth.
    Provides LOO-CV for honest generalisation estimates.
    """

    def __init__(self, jury_scores: Dict[str, float]):
        """
        Args:
            jury_scores: dict mapping repo key → jury average originality
        """
        self.jury = jury_scores

    # ── core metrics ────────────────────────────────────────────────

    def mae(self, predictions: Dict[str, float]) -> float:
        errors = self._errors(predictions)
        return float(np.mean(np.abs(errors))) if errors else float("inf")

    def rmse(self, predictions: Dict[str, float]) -> float:
        errors = self._errors(predictions)
        return float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else float("inf")

    def r_squared(self, predictions: Dict[str, float]) -> float:
        paired = self._paired(predictions)
        if not paired:
            return 0.0
        y_true = np.array([p[0] for p in paired])
        y_pred = np.array([p[1] for p in paired])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))

    # ── leave-one-out cross-validation ──────────────────────────────

    def loo_cv(self, predict_fn) -> Dict:
        """
        Leave-One-Out CV.

        Args:
            predict_fn: callable(jury_scores_minus_one) → predictions dict
                        receives the jury dict with one entry removed,
                        must return predictions for ALL repos.

        Returns:
            dict with 'mae', 'rmse', 'per_repo' results.
        """
        repos = list(self.jury.keys())
        loo_errors = []
        per_repo = []

        for held_out in repos:
            # Training jury: remove one entry
            train_jury = {k: v for k, v in self.jury.items()
                          if k != held_out}

            # Get predictions trained without `held_out`
            preds = predict_fn(train_jury)

            # Evaluate on held-out repo
            pred_val = preds.get(held_out, 0.65)
            true_val = self.jury[held_out]
            error = abs(pred_val - true_val)
            loo_errors.append(error)
            per_repo.append({
                "repo":      held_out,
                "true":      true_val,
                "predicted": pred_val,
                "error":     error,
            })

        loo_errors = np.array(loo_errors)
        return {
            "loo_mae":  float(np.mean(loo_errors)),
            "loo_rmse": float(np.sqrt(np.mean(loo_errors ** 2))),
            "loo_std":  float(np.std(loo_errors)),
            "per_repo": sorted(per_repo, key=lambda x: -x["error"]),
        }

    # ── full evaluation report ───────────────────────────────────────

    def full_report(self, predictions: Dict[str, float],
                    model_name: str = "ORACLE 2.0") -> str:
        """Generate a human-readable evaluation report."""
        paired = self._paired(predictions)
        if not paired:
            return "No jury-scored repos found in predictions."

        y_true = np.array([p[0] for p in paired])
        y_pred = np.array([p[1] for p in paired])
        errors = np.abs(y_true - y_pred)

        lines = [
            f"{'='*55}",
            f"  {model_name} — Evaluation Report",
            f"{'='*55}",
            f"  Jury-scored repos evaluated : {len(paired)}",
            f"  MAE                         : {np.mean(errors):.6f}",
            f"  RMSE                        : {np.sqrt(np.mean(errors**2)):.6f}",
            f"  R²                          : {self.r_squared(predictions):.4f}",
            f"  Max error                   : {np.max(errors):.6f}",
            f"  Min error                   : {np.min(errors):.6f}",
            f"  Std error                   : {np.std(errors):.6f}",
            f"",
            f"  Per-repo breakdown (sorted by error):",
            f"  {'Repo':<45} {'True':>6} {'Pred':>6} {'Err':>6}",
            f"  {'-'*65}",
        ]

        repo_data = sorted(
            [(self.jury[r], predictions.get(r, 0.65), r)
             for r in self.jury if r in predictions or True],
            key=lambda x: -abs(x[0] - x[1])
        )
        for true, pred, repo in repo_data:
            err = abs(true - pred)
            lines.append(
                f"  {repo:<45} {true:>6.3f} {pred:>6.3f} {err:>6.4f}"
            )

        lines.append(f"{'='*55}")
        return "\n".join(lines)

    def to_csv(self, predictions: Dict[str, float],
               path: str) -> None:
        """Save evaluation results to CSV."""
        rows = [["repo", "jury_truth", "predicted", "absolute_error"]]
        for repo, truth in self.jury.items():
            pred = predictions.get(repo, 0.65)
            rows.append([repo, truth, pred, abs(truth - pred)])
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    # ── helpers ─────────────────────────────────────────────────────

    def _errors(self, preds: Dict[str, float]) -> List[float]:
        return [abs(self.jury[r] - preds.get(r, 0.65))
                for r in self.jury]

    def _paired(self, preds: Dict[str, float]) -> List[Tuple[float, float]]:
        return [(self.jury[r], preds.get(r, 0.65))
                for r in self.jury]
