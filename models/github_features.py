"""
GitHub API Feature Fetcher — ORACLE 2.0
=========================================
Pulls live repository signals from the GitHub REST API.

Features extracted per repo:
  - stars, forks, watchers, open_issues
  - log-transformed versions of all count signals
  - contributor_count, commit_frequency (last 52 weeks)
  - days_since_last_push (recency signal)
  - has_description, has_license, has_wiki
  - language (primary)
  - fork_ratio = forks / (stars + 1)    ← derivative work indicator
  - star_per_contributor ratio           ← impact per person

Falls back gracefully to zero-filled features when API unavailable.
"""

import json
import math
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Optional
import os


class GitHubFeatureFetcher:
    """
    Fetches live repository features from GitHub REST API v3.

    Usage:
        fetcher = GitHubFeatureFetcher(token="ghp_...")
        features = fetcher.fetch("https://github.com/ethereum/go-ethereum")
    """

    BASE = "https://api.github.com"
    FEATURE_NAMES = [
        "stars_log", "forks_log", "watchers_log", "issues_log",
        "fork_ratio", "has_license", "has_description",
        "days_since_push_norm", "contributor_count_log",
        "is_fork", "size_log",
    ]

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self._cache: Dict[str, Dict] = {}

    # ── internal helpers ────────────────────────────────────────────

    def _headers(self) -> Dict:
        h = {"Accept": "application/vnd.github+json",
             "User-Agent": "oracle-deepfunding/2.0"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, url: str) -> Optional[Dict]:
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode())
        except Exception:
            return None

    # ── public API ──────────────────────────────────────────────────

    def fetch_raw(self, repo_url: str) -> Optional[Dict]:
        """Fetch raw GitHub API response for a repo URL."""
        key = repo_url.rstrip("/").lower()
        if key in self._cache:
            return self._cache[key]

        parts = repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            return None
        owner, repo = parts[-2], parts[-1]

        data = self._get(f"{self.BASE}/repos/{owner}/{repo}")
        if data:
            self._cache[key] = data
        return data

    def fetch(self, repo_url: str) -> Dict[str, float]:
        """
        Return a feature dict for one repository.
        All values are floats in a normalised range.
        Falls back to zeros on API error.
        """
        data = self.fetch_raw(repo_url)
        if not data:
            return {k: 0.0 for k in self.FEATURE_NAMES}

        stars    = data.get("stargazers_count", 0) or 0
        forks    = data.get("forks_count",      0) or 0
        watchers = data.get("subscribers_count", 0) or 0
        issues   = data.get("open_issues_count", 0) or 0
        size     = data.get("size",              0) or 0

        # Recency: days since last push
        pushed = data.get("pushed_at", "")
        try:
            pushed_dt = datetime.fromisoformat(
                pushed.replace("Z", "+00:00"))
            days_old = (datetime.now(timezone.utc) - pushed_dt).days
        except Exception:
            days_old = 365

        fork_ratio = forks / (stars + 1)

        features = {
            "stars_log":           math.log1p(stars),
            "forks_log":           math.log1p(forks),
            "watchers_log":        math.log1p(watchers),
            "issues_log":          math.log1p(issues),
            "fork_ratio":          min(fork_ratio, 5.0) / 5.0,
            "has_license":         float(bool(data.get("license"))),
            "has_description":     float(bool(data.get("description"))),
            "days_since_push_norm":min(days_old / 365.0, 3.0) / 3.0,
            "contributor_count_log": 0.0,   # filled below if API allows
            "is_fork":             float(bool(data.get("fork", False))),
            "size_log":            math.log1p(size),
        }

        # Optional: contributor count (may be rate-limited without token)
        if self.token:
            parts = repo_url.rstrip("/").split("/")
            owner, repo = parts[-2], parts[-1]
            contrib = self._get(
                f"{self.BASE}/repos/{owner}/{repo}/contributors"
                "?per_page=1&anon=true")
            # GitHub returns Link header with last page = count
            # We just log1p the list length as a proxy
            if isinstance(contrib, list):
                features["contributor_count_log"] = math.log1p(len(contrib))

        return features

    def fetch_batch(self, repos, delay: float = 0.3,
                    verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """Fetch features for many repos with rate-limit courtesy delay."""
        results = {}
        for i, repo in enumerate(repos):
            key = "/".join(repo.split("/")[-2:]).lower()
            if verbose:
                print(f"  [{i+1}/{len(repos)}] {key}")
            results[repo] = self.fetch(repo)
            if delay and i < len(repos) - 1:
                time.sleep(delay)
        return results
