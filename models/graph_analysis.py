"""
Dependency Graph Analyser — ORACLE 2.0
========================================
Parses the REAL Deep Funding dependency graph from:
https://github.com/deepfunding/dependency-graph

Graph format: {repo_url: {dep_url: weight, ...}, ...}

Features extracted:
  - weighted_in_degree  : sum of weights from repos that depend on this one
  - weighted_out_degree : sum of dependency weights going out
  - in_degree           : count of repos that depend on this
  - graph_originality   : composite originality signal from graph structure
"""

import json, csv, math
from typing import Dict, Optional


class DependencyGraphAnalyser:

    def __init__(self):
        self.graph: Dict[str, Dict[str, float]] = {}
        self.repos = []

    def load_json(self, path: str):
        """Load real Deep Funding dependency graph JSON."""
        with open(path) as f:
            self.graph = json.load(f)
        self.repos = list(self.graph.keys())
        print(f"  Loaded graph: {len(self.graph)} seed repos")

    def load_precomputed(self, path: str):
        """Load precomputed graph_features.csv."""
        self._features = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                self._features[row['repo']] = {
                    'in_degree':    float(row['in_degree']),
                    'weighted_in':  float(row['weighted_in']),
                    'out_degree':   float(row['out_degree']),
                    'weighted_out': float(row['weighted_out']),
                }

    def compute_features(self, seed_repos) -> Dict[str, Dict[str, float]]:
        """Compute graph originality features for seed repos."""
        in_deg   = {r: 0     for r in seed_repos}
        w_in     = {r: 0.0   for r in seed_repos}
        out_deg  = {r: 0     for r in seed_repos}
        w_out    = {r: 0.0   for r in seed_repos}

        repo_index = {r.lower().rstrip('/'): r for r in seed_repos}

        for src, deps in self.graph.items():
            src_key = src.lower().rstrip('/')
            if src_key in repo_index:
                r = repo_index[src_key]
                out_deg[r] = len(deps)
                w_out[r]   = sum(deps.values())

            for dep_url, weight in deps.items():
                dep_key = dep_url.lower().rstrip('/')
                if dep_key in repo_index:
                    r = repo_index[dep_key]
                    in_deg[r] += 1
                    w_in[r]   += weight

        max_win  = max(w_in.values())  or 1.0
        max_wout = max(w_out.values()) or 1.0

        features = {}
        for r in seed_repos:
            # Originality signal:
            # High weighted_in  → many depend on you → foundational → high originality
            # High weighted_out → you depend on many → integrator  → lower originality
            g_orig = (0.60 * (w_in[r]  / max_win) +
                      0.40 * (1 - min(w_out[r] / max_wout, 1)))
            features[r] = {
                'in_degree':        in_deg[r],
                'weighted_in':      w_in[r],
                'out_degree':       out_deg[r],
                'weighted_out':     w_out[r],
                'graph_originality': float(g_orig),
                'log_weighted_in':  math.log1p(w_in[r]),
            }
        return features

    def get_features(self, repo_url: str) -> Dict[str, float]:
        if hasattr(self, '_features') and repo_url in self._features:
            d = self._features[repo_url]
            max_win = max(v['weighted_in'] for v in self._features.values()) or 1
            max_wout = max(v['weighted_out'] for v in self._features.values()) or 1
            g = (0.60 * d['weighted_in']/max_win +
                 0.40 * (1 - min(d['weighted_out']/max_wout, 1)))
            return {**d, 'graph_originality': g}
        return {'in_degree':0,'weighted_in':0.0,'out_degree':0,
                'weighted_out':0.0,'graph_originality':0.45}
