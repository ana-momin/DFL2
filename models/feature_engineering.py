"""
Feature Engineering Module for ORACLE
=====================================
Extracts rich features from repository metadata for originality prediction.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RepoFeatures:
    """Structured feature set for a single repository."""
    repo_url: str
    org: str
    name: str
    
    # Tier features
    tier: str
    tier_score: float
    
    # Structural features
    is_ethereum_org: bool
    is_argotorg: bool
    is_client: bool
    is_zk: bool
    is_tooling: bool
    is_wrapper: bool
    is_research: bool
    
    # Name-based features
    has_eth_prefix: bool
    has_js_suffix: bool
    has_rs_suffix: bool
    has_py_suffix: bool
    
    # Manual originality signal
    manual_score: float


class RepositoryFeatureExtractor:
    """
    Extracts features from repository URLs and metadata.
    These features train the covariate-assisted Bradley-Terry model.
    """
    
    # Organizations known for highly original work
    HIGH_ORIGINALITY_ORGS = {
        'ethereum', 'argotorg', 'sigp', 'paradigmxyz', 'erigontech',
        'espressosystems', 'plonky3', 'powdr-labs', 'succinctlabs',
        'risc0', '0xmiden', 'consensys', 'skalenetwork'
    }
    
    # Organizations that primarily implement/wrap others' work
    LOWER_ORIGINALITY_ORGS = {
        'ethers-io', 'wevm', 'nethereum', 'hyperledger-web3j',
        'lfdt-web3j', 'wighawag', 'aestus-relay', 'smartcontracts'
    }
    
    # Repo names strongly associated with high originality
    HIGH_ORIGINALITY_NAMES = {
        'go-ethereum', 'reth', 'lighthouse', 'erigon', 'prysm',
        'nimbus-eth2', 'lodestar', 'grandine', 'nethermind', 'besu',
        'silkworm', 'evmone', 'solidity', 'vyper', 'fe', 'hevm',
        'sp1', 'miden-vm', 'plonky3', 'jellyfish', 'gnark-crypto',
        'blst', 'mcl', 'snark-verifier', 'foundry', 'halmos'
    }
    
    # Keywords indicating wrapper/integration repos (lower originality)
    WRAPPER_INDICATORS = [
        'adapter', 'wrapper', 'bridge', 'relay', 'deploy',
        'helm', 'docker', 'config', 'template', 'scaffold',
        'list', 'registry', 'chainlist'
    ]
    
    # Keywords indicating research/novel work (higher originality)
    RESEARCH_INDICATORS = [
        'zk', 'snark', 'stark', 'proof', 'prover', 'vm',
        'compiler', 'lang', 'crypto', 'consensus', 'execution',
        'verifier', 'circuit', 'polynomial', 'commitment'
    ]
    
    def extract(self, repo_url: str) -> Dict[str, float]:
        """Extract numerical features from a repository URL."""
        parts = repo_url.rstrip('/').split('/')
        org = parts[-2].lower() if len(parts) >= 2 else ''
        name = parts[-1].lower() if len(parts) >= 1 else ''
        
        features = {
            # Organization signals
            'is_high_originality_org': float(org in self.HIGH_ORIGINALITY_ORGS),
            'is_lower_originality_org': float(org in self.LOWER_ORIGINALITY_ORGS),
            'is_ethereum_org': float(org == 'ethereum'),
            'is_argotorg': float(org == 'argotorg'),
            'is_lambdaclass': float(org == 'lambdaclass'),
            'is_flashbots': float(org == 'flashbots'),
            'is_ethpandaops': float(org == 'ethpandaops'),
            
            # Name signals
            'is_high_originality_name': float(name in self.HIGH_ORIGINALITY_NAMES),
            'has_wrapper_keyword': float(any(w in name for w in self.WRAPPER_INDICATORS)),
            'has_research_keyword': float(any(r in name for r in self.RESEARCH_INDICATORS)),
            
            # Language/ecosystem signals
            'is_rust_repo': float(name.endswith('-rs') or name.endswith('rs') or 'rust' in name),
            'is_js_repo': float(name.endswith('.js') or name.endswith('-js') or 'js' in name),
            'is_python_repo': float(name.endswith('.py') or name.endswith('-py') or 'py' in name),
            
            # Ethereum client signals
            'is_execution_client': float(name in ['go-ethereum', 'reth', 'erigon', 
                                                    'nethermind', 'besu', 'silkworm',
                                                    'ethrex']),
            'is_consensus_client': float(name in ['lighthouse', 'prysm', 'teku',
                                                    'nimbus-eth2', 'lodestar', 'grandine',
                                                    'lambda-ethereum-consensus']),
            
            # ZK/Research signals  
            'is_zk_system': float(any(z in name for z in ['zk', 'snark', 'plonk', 
                                                             'stark', 'proof', 'prover'])),
            
            # Length features (longer = more specific = potentially more original)
            'org_name_length': min(len(org) / 20.0, 1.0),
            'repo_name_length': min(len(name) / 30.0, 1.0),
        }
        
        return features
    
    def extract_batch(self, repos: List[str]) -> pd.DataFrame:
        """Extract features for a list of repositories."""
        records = []
        for repo in repos:
            features = self.extract(repo)
            features['repo'] = repo
            records.append(features)
        
        return pd.DataFrame(records).set_index('repo')
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        sample = self.extract('https://github.com/ethereum/go-ethereum')
        return list(sample.keys())
