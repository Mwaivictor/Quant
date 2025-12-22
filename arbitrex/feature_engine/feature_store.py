"""
Feature Store

Immutable storage for computed feature vectors.

Rules:
    - Features never recomputed
    - Versioned by config hash
    - Same storage for backtest + live
    - Full auditability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import json
import logging

from arbitrex.feature_engine.schemas import FeatureVector, FeatureMetadata

LOG = logging.getLogger(__name__)


class FeatureStore:
    """
    Persistent storage for feature vectors.
    
    Storage Structure:
        features/
            {symbol}/
                {timeframe}/
                    {config_hash}/
                        features.parquet
                        metadata.json
    
    Guarantees:
        - Immutable once written
        - Version controlled by config hash
        - Identical access for backtest/live
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        LOG.info(f"Feature Store initialized at {self.base_path}")
    
    def write_features(
        self,
        df: pd.DataFrame,
        metadata: FeatureMetadata,
        symbol: str,
        timeframe: str,
        config_hash: str
    ):
        """
        Write feature DataFrame to store.
        
        Args:
            df: Feature DataFrame (with normalized features)
            metadata: Feature metadata
            symbol: Symbol
            timeframe: Timeframe
            config_hash: Configuration hash for versioning
        
        Storage:
            - Parquet for features (efficient, typed)
            - JSON for metadata (human-readable)
        """
        # Create directory structure
        feature_dir = self.base_path / symbol / timeframe / config_hash
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Write features to parquet
        feature_path = feature_dir / 'features.parquet'
        df.to_parquet(
            feature_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Write metadata to JSON
        metadata_path = feature_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        LOG.info(f"✓ Features written: {symbol} {timeframe} "
                f"({len(df)} bars, config {config_hash[:8]})")
    
    def read_features(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str
    ) -> Optional[pd.DataFrame]:
        """
        Read features from store.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            config_hash: Configuration hash
        
        Returns:
            Feature DataFrame or None if not found
        """
        feature_path = self.base_path / symbol / timeframe / config_hash / 'features.parquet'
        
        if not feature_path.exists():
            LOG.warning(f"Features not found: {symbol} {timeframe} config {config_hash[:8]}")
            return None
        
        df = pd.read_parquet(feature_path, engine='pyarrow')
        
        LOG.info(f"✓ Features loaded: {symbol} {timeframe} "
                f"({len(df)} bars, config {config_hash[:8]})")
        
        return df
    
    def read_metadata(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str
    ) -> Optional[Dict]:
        """Read feature metadata"""
        metadata_path = self.base_path / symbol / timeframe / config_hash / 'metadata.json'
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def exists(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str
    ) -> bool:
        """Check if features exist for given configuration"""
        feature_path = self.base_path / symbol / timeframe / config_hash / 'features.parquet'
        return feature_path.exists()
    
    def list_versions(
        self,
        symbol: str,
        timeframe: str
    ) -> List[str]:
        """List all config versions available for symbol/timeframe"""
        version_dir = self.base_path / symbol / timeframe
        
        if not version_dir.exists():
            return []
        
        versions = [d.name for d in version_dir.iterdir() if d.is_dir()]
        return sorted(versions)
    
    def get_latest_features(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Get features from most recent configuration"""
        versions = self.list_versions(symbol, timeframe)
        
        if not versions:
            return None
        
        # Use most recent version
        latest_version = versions[-1]
        
        return self.read_features(symbol, timeframe, latest_version)
