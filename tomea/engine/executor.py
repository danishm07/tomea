import modal
import os
import sys
from typing import Dict, Any, Generator

# 1. Define the Cloud Environment
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "transformers", "datasets", "pandas", "scikit-learn",
        "accelerate", "numpy<2.0.0", "rich", "asciichartpy", "peft"
    )
    .add_local_dir("cache", remote_path="/root/cache")
)

app = modal.App("tomeav3")

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    cpu=2.0,
    memory=4096
)
def run_harness_with_adapter(
    harness_code: str, 
    adapter_code: str, 
    dataset_url: str, 
    method_name: str
) -> Generator[str, None, Dict[str, Any]]:
    """
    Generator function that yields log lines in real-time,
    and yields the final result dict as the last item.
    """
    import sys
    import subprocess
    import importlib.util
    import time
    
    print(f"--- Setting up environment for {method_name} ---")
    
    # 1. Write Adapter
    with open("adapter.py", "w") as f:
        f.write(adapter_code)

    # 2. Write Harness
    with open("harness_run.py", "w") as f:
        f.write(harness_code)
        # Append the actual execution call to the script
        f.write(f"\n\nif __name__ == '__main__':\n    run_experiment('{dataset_url}', '{method_name}')")

    # 3. Run as Subprocess to capture output in real-time
    # This is safer than hijacking sys.stdout for streaming
    process = subprocess.Popen(
        [sys.executable, "-u", "harness_run.py"], # -u for unbuffered stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1 # Line buffered
    )

    # 4. Stream Logs
    captured_logs = []
    
    # Yield logs as they come
    for line in process.stdout:
        line = line.strip()
        if line:
            captured_logs.append(line)
            yield f"LOG:{line}" # Yield string to client

    # 5. Wait for finish
    return_code = process.wait()
    
    if return_code != 0:
        yield {
            "status": "failed",
            "error": f"Process exited with code {return_code}",
            "logs": "\n".join(captured_logs)
        }
        return

    # 6. Retrieve Results (Harness saves metrics.json)
    import json
    try:
        with open("/root/experiment/results/metrics.json", "r") as f:
            metrics = json.load(f)
        metrics["logs"] = "\n".join(captured_logs)
        yield metrics # Yield dict as final result
    except Exception as e:
        yield {
            "status": "failed",
            "error": f"Could not read metrics.json: {e}",
            "logs": "\n".join(captured_logs)
        }

class ModalExecutor:
    """
    Local client that triggers the remote function and handles the stream.
    """
    def run_experiment(
        self, 
        harness_code: str, 
        adapter_code: str, 
        dataset_url: str, 
        method_name: str,
        log_callback=None
    ) -> Dict[str, Any]:
        
        full_logs = []
        final_result = {"status": "failed", "error": "No result yielded"}

        try:
            # Call remote generator
            for update in run_harness_with_adapter.remote_gen(
                harness_code=harness_code,
                adapter_code=adapter_code,
                dataset_url=dataset_url,
                method_name=method_name
            ):
                # Check type of update
                if isinstance(update, str):
                    # It's a log line
                    clean_log = update.replace("LOG:", "")
                    full_logs.append(clean_log)
                    if log_callback:
                        log_callback(clean_log)
                elif isinstance(update, dict):
                    # It's the final result
                    final_result = update
            
            return final_result
            
        except Exception as e:
            return {"status": "failed", "error": f"Modal Connection Error: {str(e)}", "logs": "\n".join(full_logs)}