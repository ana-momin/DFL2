"""
Dependency Graph Analyser — ORACLE 2.0
========================================
Computes graph-theoretic originality signals from the Ethereum
dependency graph provided by Deep Funding.

Key insight: repos that MANY others depend on are foundational
(high originality); repos that depend on many others are integrators
(lower originality).

Features computed:
  - reverse_pagerank      : how important is this node to its dependents?
  - in_degree_norm        : number of repos that directly depend on this one
  - out_degree_norm       : number of dependencies this repo has
  - betweenness_norm      : bridge repos (potentially lower originality)
  - dependency_depth      : how deep in the stack is this repo?
  - is_root               : has dependents but no dependencies?
"""

import math
import csv
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque


class DependencyGraphAnalyser:
    """
    Builds and analyses the Ethereum dependency graph.

    The graph is directed: edge A → B means "A depends on B"
    (B is a dependency of A).

    For originality:
      - High in-degree (many depend on you)  → high originality
      - High out-degree (you depend on many) → lower originality
      - Root nodes (no outgoing edges)       → highest originality
    """

    def __init__(self):
        # Adjacency: dependents[B] = {A, C, ...} — repos that need B
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        # Adjacency: dependencies[A] = {B, C, ...} — what A needs
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.all_nodes: Set[str] = set()

    # ── graph construction ──────────────────────────────────────────

    def add_edge(self, dependent: str, dependency: str):
        """Record that `dependent` depends on `dependency`."""
        d = dependent.lower().strip()
        p = dependency.lower().strip()
        self.dependents[p].add(d)
        self.dependencies[d].add(p)
        self.all_nodes.update([d, p])

    def add_repo(self, repo: str):
        """Ensure a repo appears in the graph even if it has no edges."""
        self.all_nodes.add(repo.lower().strip())

    def load_from_csv(self, path: str,
                      dependent_col: str = "dependent",
                      dependency_col: str = "dependency"):
        """Load edges from a CSV file."""
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dep_on = row.get(dependent_col, "").strip()
                    dep_of = row.get(dependency_col, "").strip()
                    if dep_on and dep_of:
                        self.add_edge(dep_on, dep_of)
            print(f"  Loaded {len(self.all_nodes)} nodes, "
                  f"{sum(len(v) for v in self.dependents.values())} edges")
        except FileNotFoundError:
            print(f"  Warning: dependency graph file not found: {path}")

    # ── graph algorithms ────────────────────────────────────────────

    def _pagerank(self, damping: float = 0.85,
                  max_iter: int = 100,
                  tol: float = 1e-6) -> Dict[str, float]:
        """
        Reverse PageRank: authority flows from dependents to dependencies.
        A repo that many important repos depend on gets a high score.
        """
        nodes = list(self.all_nodes)
        n = len(nodes)
        if n == 0:
            return {}

        rank = {node: 1.0 / n for node in nodes}
        out_count = {node: len(self.dependencies.get(node, set()))
                     for node in nodes}

        for _ in range(max_iter):
            new_rank = {}
            for node in nodes:
                # Collect rank from nodes that depend on `node`
                incoming = 0.0
                for dep in self.dependents.get(node, set()):
                    oc = out_count.get(dep, 0)
                    if oc > 0:
                        incoming += rank[dep] / oc

                new_rank[node] = (1 - damping) / n + damping * incoming

            # Check convergence
            delta = sum(abs(new_rank[n] - rank[n]) for n in nodes)
            rank = new_rank
            if delta < tol:
                break

        return rank

    def _bfs_depth(self, start: str) -> int:
        """BFS to find the maximum distance to a root (no dependencies)."""
        visited = {start}
        queue = deque([(start, 0)])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            for dep in self.dependencies.get(node, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append((dep, depth + 1))
        return max_depth

    # ── feature extraction ──────────────────────────────────────────

    def compute_features(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all graph features for every node.
        Returns dict: repo_key → feature_dict
        """
        if not self.all_nodes:
            return {}

        print("  Computing PageRank...")
        pr = self._pagerank()
        max_pr = max(pr.values()) if pr else 1.0

        max_in  = max((len(self.dependents.get(n, set()))
                       for n in self.all_nodes), default=1)
        max_out = max((len(self.dependencies.get(n, set()))
                       for n in self.all_nodes), default=1)

        features = {}
        for node in self.all_nodes:
            in_deg  = len(self.dependents.get(node, set()))
            out_deg = len(self.dependencies.get(node, set()))

            is_root = (out_deg == 0 and in_deg > 0)
            is_leaf = (in_deg == 0)

            # Originality signal from graph structure:
            # Root nodes (depended on by many, depend on nothing) → high
            # Leaf nodes (depend on others, nobody depends on them) → low
            graph_originality = (
                0.5 * (in_deg / max(max_in, 1)) +           # in-degree
                0.3 * (pr.get(node, 0) / max(max_pr, 1e-9)) +  # pagerank
                0.2 * (1.0 - out_deg / max(max_out, 1))     # inv out-degree
            )

            features[node] = {
                "pagerank_norm":     pr.get(node, 0.0) / max(max_pr, 1e-9),
                "in_degree_norm":    in_deg / max(max_in, 1),
                "out_degree_norm":   out_deg / max(max_out, 1),
                "is_root":           float(is_root),
                "is_leaf":           float(is_leaf),
                "graph_originality": graph_originality,
                "log_in_degree":     math.log1p(in_deg),
                "log_out_degree":    math.log1p(out_deg),
            }

        return features

    def get_features_for_repo(self, repo_url: str,
                               all_features: Optional[Dict] = None
                               ) -> Dict[str, float]:
        """Get graph features for a specific repo URL."""
        if all_features is None:
            all_features = self.compute_features()

        key = "/".join(repo_url.split("/")[-2:]).lower()
        return all_features.get(key, {
            "pagerank_norm":     0.0,
            "in_degree_norm":    0.0,
            "out_degree_norm":   0.0,
            "is_root":           0.0,
            "is_leaf":           1.0,
            "graph_originality": 0.5,
            "log_in_degree":     0.0,
            "log_out_degree":    0.0,
        })

    def summary(self) -> str:
        """Return a text summary of the graph."""
        n_nodes = len(self.all_nodes)
        n_edges = sum(len(v) for v in self.dependents.values())
        roots = sum(1 for n in self.all_nodes
                    if not self.dependencies.get(n) and self.dependents.get(n))
        return (f"Graph: {n_nodes} nodes, {n_edges} edges, "
                f"{roots} root nodes (pure dependencies)")
