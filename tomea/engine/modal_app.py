import modal
import sys
import os 



#defining docker image-> standard 
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate", 
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "seaborn",
        "bitsandbytes",
        "peft",
        "huggingface_hub",
        "rich",
        "langchain-openai",
        "langchain-community",
        "python-dotenv",
        "asciichartpy",
        "pymupdf",
        "arxiv",
        "bs4",
        "evaluate",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("tomeav3")

#defining gpu function 
@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    cpu=4,
    memory=16384,
)
def run_validation_job(code_string: str, dataset_url: str = None):
    """Streams stdout/stderr line-by-line in real-time"""
    import subprocess
    import select
    
    os.makedirs("/root/experiment", exist_ok=True)
    os.chdir("/root/experiment")
    
    script_name = "train_script.py"
    with open(script_name, "w") as f:
        f.write(code_string)
    
    env = os.environ.copy()
    if dataset_url:
        env["USER_DATASET_URL"] = dataset_url
    
    # Start process with merged output
    process = subprocess.Popen(
        [sys.executable, "-u", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # CRITICAL: Merge stderr into stdout
        text=True,
        env=env,
        bufsize=1
    )
    
    # Yield lines as they come
    for line in iter(process.stdout.readline, ''):
        if line:
            yield {"type": "stdout", "line": line.rstrip()}
    
    # Wait for completion
    process.wait()
    
    # Send final status
    yield {
        "type": "complete",
        "exit_code": process.returncode,
        "status": "success" if process.returncode == 0 else "failed"
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    cpu=4,
    memory=16384,
)
def run_validation_job_stream(code_string: str, dataset_url: str = None):
    """Streaming version - yields output line-by-line"""
    import subprocess
    
    print("Starting validation job on Modal (streaming mode)")
    
    # Setup
    os.makedirs("/root/experiment", exist_ok=True)
    os.chdir("/root/experiment")
    
    script_name = "train_script.py"
    with open(script_name, "w") as f:
        f.write(code_string)
    
    env = os.environ.copy()
    if dataset_url:
        env["USER_DATASET_URL"] = dataset_url
    
    # Start process
    process = subprocess.Popen(
        [sys.executable, "-u", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        env=env,
        bufsize=1  # Line buffered
    )
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            yield {"type": "stdout", "line": line.rstrip()}
    
    # Wait for completion
    process.wait()
    
    # Send final status
    yield {
        "type": "complete",
        "exit_code": process.returncode,
        "status": "success" if process.returncode == 0 else "failed"
    }

# KEEP everything as-is, but ADD this at the end:

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    cpu=4,
    memory=16384,
)
def run_harness_with_adapter(
    harness_code: str,
    adapter_code: str,
    dataset_url: str,
    method_name: str = "paper_method"
):
    """
    Run golden harness with generated adapter.
    
    This is the NEW preferred way to run experiments.
    The old run_validation_job is kept for backwards compatibility.
    """
    import subprocess
    import json
    
    print(f"Starting harness execution for {method_name}")
    
    # Setup
    os.makedirs("/root/experiment", exist_ok=True)
    os.chdir("/root/experiment")
    
    # Write adapter
    with open("adapter.py", "w") as f:
        f.write(adapter_code)
    
    # Write harness
    with open("harness.py", "w") as f:
        f.write(harness_code)
    
    # Run harness
    env = os.environ.copy()
    env["USER_DATASET_URL"] = dataset_url
    
    process = subprocess.Popen(
        [sys.executable, "-u", "harness.py",
         "--dataset", dataset_url,
         "--method", method_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1
    )
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            yield {"type": "stdout", "line": line.rstrip()}
    
    process.wait()
    
    # Read metrics file
    try:
        with open("/root/experiment/results/metrics.json") as f:
            metrics = json.load(f)
        
        yield {
            "type": "complete",
            "exit_code": process.returncode,
            "status": metrics.get('status', 'success'),
            "metrics": metrics
        }
    except Exception as e:
        yield {
            "type": "complete",
            "exit_code": process.returncode,
            "status": "failed",
            "error": str(e)
        }


"""""

@app.local_entrypoint()
def main():
    print("testing connection to modal:....")
    test_code = 
import torch
import os
print(f"Running on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Dataset URL env var: {os.getenv('USER_DATASET_URL', 'Not Set')}")

# Simulate a tiny training loop
for i in range(3):
    print(f"Step {i}: Training...")


    
    dataset = "https://example.com/data.csv"
    
    # This triggers the remote execution
    print("   Sending job to remote GPU (this might take a minute to boot first time)...")
    res = run_validation_job.remote(test_code, dataset)
    
    print("\n--- RESULT ---")
    print(f"Status: {res['status']}")
    print("STDOUT:", res['stdout'])
    print("STDERR:", res['stderr'])

"""""