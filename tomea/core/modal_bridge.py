import modal
import os
import sys
import asyncio
from typing import Tuple

# Standard V2 Image with all necessary libraries
tomea_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Core ML stack
        "torch", "transformers", "datasets", "accelerate", "pandas", 
        "numpy", "scikit-learn", "tqdm", "matplotlib", "seaborn",
        "bitsandbytes", "peft", "huggingface_hub", "rich", 
        "python-dotenv", "asciichartpy", "trl", "einops",
        # Scientific computing (shared dep for finance + bio)
        "scipy",
        # Protocol 1: Quantitative Finance
        # NOTE: pandas_ta is NOT installed here. It requires Python >=3.11
        # and bloats every image build even when not needed. It is lazy-installed
        # inside experiment.py only when the synthesizer detects a finance paper
        # (via extraction_metadata["required_deps"]). statsmodels and yfinance
        # are safe to keep — they build on all supported versions.
        "statsmodels",   # Time-series analysis, rolling stats
        "yfinance",      # Historical price data download
        # Protocol 2: Biomedical
        # NOTE: pyhealth is NOT included here. It requires MIMIC-III access
        # credentials and will fail in a sandboxed container. Protocol 2
        # should use a HF bio dataset (e.g., "bigbio/pubmed_qa") instead,
        # or we need to pre-download the data and mount it.
        # "pyhealth",  
    )
)

app = modal.App("tomea-v2-execution")

@app.function(
    image=tomea_image,
    gpu="A10G", 
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_remote_experiment(code: str):
    import subprocess
    import sys
    
    print("--- ☁️ Cloud Environment Initialized ---")
    os.makedirs("/root/experiment", exist_ok=True)
    os.chdir("/root/experiment")
    
    with open("experiment.py", "w") as f:
        f.write(code)
    
    # Run unbuffered
    process = subprocess.Popen(
        [sys.executable, "-u", "experiment.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ''):
        if line: yield line.strip()
            
    process.wait()
    if process.returncode == 0:
        yield "SUCCESS_MARKER"
    else:
        yield f"FAILURE_MARKER_CODE_{process.returncode}"

class ModalExecutor:
    async def run(self, code: str) -> Tuple[bool, str]:
        print("   ☁️ Connecting to Modal GPU...")
        try:
            loop = asyncio.get_event_loop()
            
            def run_sync():
                with app.run():
                    logs = []
                    is_success = False
                    for log_line in run_remote_experiment.remote_gen(code):
                        if log_line == "SUCCESS_MARKER":
                            is_success = True
                        elif log_line.startswith("FAILURE_MARKER"):
                            is_success = False
                        else:
                            logs.append(log_line) # Keep RAW log
                            if not self._is_noise(log_line):
                                print(f"      [GPU] {log_line}")
                    return is_success, logs

            success, raw_log_list = await loop.run_in_executor(None, run_sync)
            return success, "\n".join(raw_log_list)

        except Exception as e:
            error = f"MODAL SYSTEM ERROR: {e}"
            print(f"      ❌ {error}")
            return False, error

    def _is_noise(self, line: str) -> bool:
        # Hide progress bars from console, but keep in file
        if "Loading weights" in line: return True
        if "Materializing param" in line: return True
        if "%|" in line and "it/s" in line: return True
        if line.strip() == "": return True
        return False