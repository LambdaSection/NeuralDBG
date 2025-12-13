from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DatasetVersion:
    def __init__(
        self,
        dataset_path: Union[str, Path],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.version = version or self._generate_version()
        self.metadata = metadata or {}
        self.tags = tags or []
        self.checksum = self._compute_checksum()
        self.created_at = datetime.now().isoformat()

    def _generate_version(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"

    def _compute_checksum(self) -> str:
        hasher = hashlib.sha256()
        
        if self.dataset_path.is_file():
            with open(self.dataset_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        elif self.dataset_path.is_dir():
            for file_path in sorted(self.dataset_path.rglob("*")):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.dataset_path)
                    hasher.update(str(relative_path).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "version": self.version,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetVersion:
        obj = cls.__new__(cls)
        obj.dataset_path = Path(data["dataset_path"])
        obj.version = data["version"]
        obj.checksum = data["checksum"]
        obj.metadata = data.get("metadata", {})
        obj.tags = data.get("tags", [])
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        return obj


class DatasetVersionManager:
    def __init__(self, base_dir: Union[str, Path] = ".neural_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_dir / "versions.json"
        self._load_versions()

    def _load_versions(self):
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                data = json.load(f)
                self.versions = {
                    k: DatasetVersion.from_dict(v) for k, v in data.items()
                }
        else:
            self.versions = {}

    def _save_versions(self):
        data = {k: v.to_dict() for k, v in self.versions.items()}
        with open(self.versions_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_version(
        self,
        dataset_path: Union[str, Path],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        copy_data: bool = True,
    ) -> DatasetVersion:
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        dataset_version = DatasetVersion(dataset_path, version, metadata, tags)
        
        if copy_data:
            version_dir = self.base_dir / "datasets" / dataset_version.version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            if dataset_path.is_file():
                shutil.copy2(dataset_path, version_dir / dataset_path.name)
                dataset_version.dataset_path = version_dir / dataset_path.name
            elif dataset_path.is_dir():
                dest_dir = version_dir / dataset_path.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(dataset_path, dest_dir)
                dataset_version.dataset_path = dest_dir

        self.versions[dataset_version.version] = dataset_version
        self._save_versions()
        
        return dataset_version

    def get_version(self, version: str) -> Optional[DatasetVersion]:
        return self.versions.get(version)

    def list_versions(
        self, tags: Optional[List[str]] = None
    ) -> List[DatasetVersion]:
        versions = list(self.versions.values())
        
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def delete_version(self, version: str) -> bool:
        if version not in self.versions:
            return False
        
        dataset_version = self.versions[version]
        version_dir = self.base_dir / "datasets" / version
        
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        del self.versions[version]
        self._save_versions()
        
        return True

    def compare_versions(
        self, version1: str, version2: str
    ) -> Dict[str, Any]:
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        return {
            "version1": version1,
            "version2": version2,
            "checksum_match": v1.checksum == v2.checksum,
            "checksum1": v1.checksum,
            "checksum2": v2.checksum,
            "metadata_diff": self._diff_metadata(v1.metadata, v2.metadata),
            "tags1": v1.tags,
            "tags2": v2.tags,
            "created_at1": v1.created_at,
            "created_at2": v2.created_at,
        }

    def _diff_metadata(
        self, meta1: Dict[str, Any], meta2: Dict[str, Any]
    ) -> Dict[str, Any]:
        all_keys = set(meta1.keys()) | set(meta2.keys())
        diff = {}
        
        for key in all_keys:
            if key not in meta1:
                diff[key] = {"status": "added", "value": meta2[key]}
            elif key not in meta2:
                diff[key] = {"status": "removed", "value": meta1[key]}
            elif meta1[key] != meta2[key]:
                diff[key] = {
                    "status": "changed",
                    "old": meta1[key],
                    "new": meta2[key],
                }
        
        return diff

    def tag_version(self, version: str, tags: List[str]) -> bool:
        if version not in self.versions:
            return False
        
        self.versions[version].tags.extend(tags)
        self.versions[version].tags = list(set(self.versions[version].tags))
        self._save_versions()
        
        return True

    def get_by_checksum(self, checksum: str) -> Optional[DatasetVersion]:
        for version in self.versions.values():
            if version.checksum == checksum:
                return version
        return None
