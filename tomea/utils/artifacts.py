"""
Artifact Manager - Persist every experiment run.
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ArtifactManager:
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_run(self, paper_name: str, method_type: str) -> Path:
        """Create a unique directory for this specific run."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize name
        safe_name = "".join(c if c.isalnum() else "_" for c in paper_name)
        run_dir = self.base_dir / f"{timestamp}_{safe_name}_{method_type}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_code(self, run_dir: Path, filename: str, code: str):
        """Save generated Python code."""
        try:
            with open(run_dir / filename, "w") as f:
                f.write(code)
        except Exception as e:
            logger.error(f"Failed to save code artifact: {e}")

    def save_metadata(self, run_dir: Path, data: Dict[str, Any]):
        """Save configuration or metrics."""
        try:
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def save_log(self, run_dir: Path, log_content: str, filename: str = "training.log"):
        """Save raw logs."""
        try:
            with open(run_dir / filename, "w") as f:
                f.write(log_content)
        except Exception as e:
            logger.error(f"Failed to save log: {e}")
            
    def save_report(self, run_dir: Path, markdown: str):
        """Save the final analysis report."""
        try:
            with open(run_dir / "report.md", "w") as f:
                f.write(markdown)
        except Exception as e:
            logger.error(f"Failed to save report: {e}")