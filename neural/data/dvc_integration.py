from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Union


class DVCIntegration:
    def __init__(self, repo_path: Union[str, Path] = "."):
        self.repo_path = Path(repo_path)
        self.dvc_dir = self.repo_path / ".dvc"
        self.is_initialized = self.dvc_dir.exists()

    def init(self) -> bool:
        if self.is_initialized:
            return True
        
        try:
            subprocess.run(
                ["dvc", "init"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            self.is_initialized = True
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def add(
        self,
        file_path: Union[str, Path],
        remote: Optional[str] = None,
    ) -> bool:
        if not self.is_initialized:
            if not self.init():
                return False
        
        try:
            file_path = Path(file_path)
            subprocess.run(
                ["dvc", "add", str(file_path)],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            if remote:
                subprocess.run(
                    ["dvc", "push", str(file_path) + ".dvc", "-r", remote],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def pull(
        self,
        file_path: Optional[Union[str, Path]] = None,
        remote: Optional[str] = None,
    ) -> bool:
        if not self.is_initialized:
            return False
        
        try:
            cmd = ["dvc", "pull"]
            
            if file_path:
                cmd.append(str(Path(file_path)) + ".dvc")
            
            if remote:
                cmd.extend(["-r", remote])
            
            subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def status(self, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        if not self.is_initialized:
            return {"error": "DVC not initialized"}
        
        try:
            cmd = ["dvc", "status"]
            
            if file_path:
                cmd.append(str(Path(file_path)) + ".dvc")
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            return {"status": result.stdout.strip() or "up to date"}
        except subprocess.CalledProcessError as e:
            return {"error": e.stderr.strip()}
        except FileNotFoundError:
            return {"error": "DVC not installed"}

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        if not self.is_initialized:
            return {"error": "DVC not initialized"}
        
        dvc_file = Path(file_path).with_suffix(Path(file_path).suffix + ".dvc")
        
        if not dvc_file.exists():
            return {"error": "DVC file not found"}
        
        try:
            with open(dvc_file, "r") as f:
                json.load(f) if dvc_file.suffix == ".json" else {}
                
                import yaml
                dvc_file_content = yaml.safe_load(open(dvc_file, "r"))
                
                return dvc_file_content
        except Exception as e:
            return {"error": str(e)}

    def add_remote(
        self,
        name: str,
        url: str,
        default: bool = False,
    ) -> bool:
        if not self.is_initialized:
            if not self.init():
                return False
        
        try:
            subprocess.run(
                ["dvc", "remote", "add", name, url],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            if default:
                subprocess.run(
                    ["dvc", "remote", "default", name],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def list_remotes(self) -> List[str]:
        if not self.is_initialized:
            return []
        
        try:
            result = subprocess.run(
                ["dvc", "remote", "list"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            remotes = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    remotes.append(line.split()[0])
            
            return remotes
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def checkout(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        if not self.is_initialized:
            return False
        
        try:
            cmd = ["dvc", "checkout"]
            
            if file_path:
                cmd.append(str(Path(file_path)) + ".dvc")
            
            subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_tracked_files(self) -> List[str]:
        if not self.is_initialized:
            return []
        
        tracked_files = []
        
        for dvc_file in self.repo_path.rglob("*.dvc"):
            original_file = dvc_file.with_suffix("")
            tracked_files.append(str(original_file.relative_to(self.repo_path)))
        
        return tracked_files

    def diff(self, rev1: str = "HEAD", rev2: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_initialized:
            return {"error": "DVC not initialized"}
        
        try:
            cmd = ["dvc", "diff", rev1]
            
            if rev2:
                cmd.append(rev2)
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            return {"diff": result.stdout.strip()}
        except subprocess.CalledProcessError as e:
            return {"error": e.stderr.strip()}
        except FileNotFoundError:
            return {"error": "DVC not installed"}

    def is_dvc_available(self) -> bool:
        try:
            subprocess.run(
                ["dvc", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
