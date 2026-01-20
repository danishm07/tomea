# utils/results_saver.py
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict


def create_run_directory(paper_name: str, base_dir: str = "runs") -> Path:
    """
    Create a unique directory for this paper run.
    
    Format: runs/2025-01-15_1843_lora/
    
    Args:
        paper_name: Name of the paper (e.g., "LoRA")
        base_dir: Base directory for all runs
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    # Sanitize paper name (remove spaces, special chars)
    safe_name = "".join(c if c.isalnum() else "_" for c in paper_name.lower())
    
    run_name = f"{timestamp}_{safe_name}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    arxiv_id: str,
    paper_name: str,
    dataset_url: str,
    generated_code: str,
    training_log: str,
    final_metrics: Dict,
    error_log: list,
    success: bool,
    attempts: int,
    wandb_url: Optional[str] = None
):
    """
    Save all artifacts from a single paper run.
    
    Args:
        run_dir: Directory to save to
        arxiv_id: ArXiv ID of paper
        paper_name: Human-readable paper name
        dataset_url: Dataset used
        generated_code: The full training script that was generated
        training_log: Complete stdout/stderr
        final_metrics: Dict with final_loss, final_eval_loss, etc.
        error_log: List of errors encountered
        success: Whether run succeeded
        attempts: Number of attempts taken
        wandb_url: Optional W&B dashboard URL
    """
    
    # 1. CONFIG.JSON - What was run
    config = {
        "arxiv_id": arxiv_id,
        "paper_name": paper_name,
        "dataset_url": dataset_url,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "attempts": attempts,
        "wandb_url": wandb_url
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 2. GENERATED_CODE.PY - The training script
    with open(run_dir / "generated_code.py", "w") as f:
        f.write(generated_code)
    
    # 3. TRAINING_LOG.TXT - Full stdout
    with open(run_dir / "training_log.txt", "w") as f:
        f.write(training_log)
    
    # 4. METRICS.JSON - Final results
    metrics = {
        **final_metrics,
        "success": success,
        "attempts": attempts
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 5. ERRORS.JSON - Any errors (even if succeeded)
    if error_log:
        with open(run_dir / "errors.json", "w") as f:
            json.dump(error_log, f, indent=2)
    
    print(f"   💾 Saved artifacts to: {run_dir}")


def get_latest_runs(base_dir: str = "runs", limit: int = 10) -> list:
    """
    Get the N most recent runs, sorted by timestamp.
    
    Returns:
        List of dicts with run metadata
    """
    runs_path = Path(base_dir)
    if not runs_path.exists():
        return []
    
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    runs = []
    for run_dir in run_dirs[:limit]:
        config_file = run_dir / "config.json"
        metrics_file = run_dir / "metrics.json"
        
        if config_file.exists() and metrics_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            runs.append({
                "run_dir": str(run_dir),
                "paper_name": config.get("paper_name"),
                "arxiv_id": config.get("arxiv_id"),
                "timestamp": config.get("timestamp"),
                "success": metrics.get("success"),
                **metrics
            })
    
    return runs