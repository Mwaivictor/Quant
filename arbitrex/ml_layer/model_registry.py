"""
Model Registry and Versioning

Manages ML model storage, loading, and version control.
"""

import os
import json
import pickle
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path
import logging

from arbitrex.ml_layer.schemas import ModelMetadata

LOG = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for ML models with versioning and governance.
    
    Ensures:
        - Model versioning (semantic versioning)
        - Model lineage tracking
        - Config hash linkage
        - Metadata storage
        - Rollback capability
    """
    
    def __init__(self, storage_path: str = "arbitrex/ml_layer/models"):
        """
        Initialize model registry.
        
        Args:
            storage_path: Path to model storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Registry index file
        self.index_file = self.storage_path / "registry_index.json"
        self.index = self._load_index()
        
        LOG.info(f"Model registry initialized at {self.storage_path}")
    
    def _load_index(self) -> Dict:
        """Load registry index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {
            'regime_classifier': [],
            'signal_filter': []
        }
    
    def _save_index(self):
        """Save registry index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(
        self,
        model_type: str,
        model,
        metadata: ModelMetadata
    ) -> str:
        """
        Register a new model.
        
        Args:
            model_type: "regime_classifier" or "signal_filter"
            model: Trained model object
            metadata: Model metadata
        
        Returns:
            Model version
        """
        # Generate version
        version = metadata.model_version
        
        # Create model directory
        model_dir = self.storage_path / model_type / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update index
        if model_type not in self.index:
            self.index[model_type] = []
        
        self.index[model_type].append({
            'version': version,
            'registered_on': datetime.now().isoformat(),
            'config_hash': metadata.config_hash,
            'model_path': str(model_path),
            'metadata_path': str(metadata_path)
        })
        
        self._save_index()
        
        LOG.info(f"Registered {model_type} model version {version}")
        
        return version
    
    def load_model(
        self,
        model_type: str,
        version: Optional[str] = None
    ) -> tuple:
        """
        Load model from registry.
        
        Args:
            model_type: "regime_classifier" or "signal_filter"
            version: Model version (uses latest if None)
        
        Returns:
            (model, metadata)
        """
        if model_type not in self.index or not self.index[model_type]:
            raise ValueError(f"No models registered for {model_type}")
        
        # Get version entry
        if version is None:
            # Use latest
            entry = self.index[model_type][-1]
        else:
            # Find specific version
            entry = next(
                (e for e in self.index[model_type] if e['version'] == version),
                None
            )
            if entry is None:
                raise ValueError(f"Model version {version} not found for {model_type}")
        
        # Load model
        model_path = Path(entry['model_path'])
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = Path(entry['metadata_path'])
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Reconstruct metadata object
        metadata = ModelMetadata(**metadata_dict)
        
        LOG.info(f"Loaded {model_type} model version {entry['version']}")
        
        return model, metadata
    
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, List]:
        """
        List registered models.
        
        Args:
            model_type: Filter by type (None = all)
        
        Returns:
            Dictionary of model listings
        """
        if model_type:
            return {model_type: self.index.get(model_type, [])}
        return self.index
    
    def get_latest_version(self, model_type: str) -> Optional[str]:
        """
        Get latest version for model type.
        
        Args:
            model_type: Model type
        
        Returns:
            Latest version or None
        """
        if model_type not in self.index or not self.index[model_type]:
            return None
        
        return self.index[model_type][-1]['version']
    
    def delete_model(self, model_type: str, version: str):
        """
        Delete model from registry.
        
        Args:
            model_type: Model type
            version: Version to delete
        """
        # Find entry
        entry = next(
            (e for e in self.index[model_type] if e['version'] == version),
            None
        )
        
        if entry is None:
            LOG.warning(f"Model {model_type} version {version} not found")
            return
        
        # Delete files
        model_path = Path(entry['model_path'])
        metadata_path = Path(entry['metadata_path'])
        
        if model_path.exists():
            model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove directory if empty
        model_dir = model_path.parent
        if model_dir.exists() and not list(model_dir.iterdir()):
            model_dir.rmdir()
        
        # Update index
        self.index[model_type] = [
            e for e in self.index[model_type]
            if e['version'] != version
        ]
        
        self._save_index()
        
        LOG.info(f"Deleted {model_type} model version {version}")


class ModelVersionManager:
    """
    Manages model version strings.
    
    Uses semantic versioning: vMAJOR.MINOR.PATCH
    """
    
    @staticmethod
    def parse_version(version: str) -> tuple:
        """
        Parse version string.
        
        Args:
            version: Version string (e.g., "v1.2.3")
        
        Returns:
            (major, minor, patch)
        """
        if version.startswith('v'):
            version = version[1:]
        
        parts = version.split('.')
        return tuple(int(p) for p in parts)
    
    @staticmethod
    def increment_version(
        current_version: str,
        level: str = 'patch'
    ) -> str:
        """
        Increment version.
        
        Args:
            current_version: Current version string
            level: 'major', 'minor', or 'patch'
        
        Returns:
            New version string
        """
        major, minor, patch = ModelVersionManager.parse_version(current_version)
        
        if level == 'major':
            major += 1
            minor = 0
            patch = 0
        elif level == 'minor':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"v{major}.{minor}.{patch}"
    
    @staticmethod
    def format_version(major: int, minor: int, patch: int) -> str:
        """Format version string"""
        return f"v{major}.{minor}.{patch}"
