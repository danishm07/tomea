import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime

class ArtifactManager:
    def __init__(self, base_path="./tomea_runs"):
        self.base_path = Path(base_path)
        self.run_id = str(uuid.uuid4())[:8]  # Short UUID
        self.run_dir = self.base_path / self.run_id
        self.src_dir = self.run_dir / "src"
        self.logs_dir = self.run_dir / "logs"
        
        # Setup directories
        self._setup_directories()
        
        # Setup internal logger
        self.logger = self._setup_logging()

    def _setup_directories(self):
        """Creates the physical folder structure."""
        self.src_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Sets up a file logger that writes to the run directory."""
        logger = logging.getLogger(f"tomea_{self.run_id}")
        logger.setLevel(logging.INFO)
        
        # File Handler (Persist to disk)
        fh = logging.FileHandler(self.logs_dir / "system.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def get_paths(self):
        """Returns paths for the Agents to use."""
        return {
            "root": str(self.run_dir),
            "src": str(self.src_dir),
            "logs": str(self.logs_dir)
        }

    def save_code(self, filename, content):
        """Standard way to save generated code."""
        # Security check: Prevent writing outside src dir
        clean_filename = os.path.basename(filename)
        file_path = self.src_dir / clean_filename
        
        with open(file_path, "w") as f:
            f.write(content)
        
        self.logger.info(f"Saved artifact: {clean_filename}")
        return str(file_path)

    def save_manifest(self, metadata):
        """Saves the final run report."""
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            **metadata
        }
        with open(self.run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)