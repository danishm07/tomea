import asyncio
import pandas as pd
import logging
import os
import requests
import traceback
import re
from typing import List, Dict
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
import time

# Imports
from tomea.ui.parallel_dashboard import ParallelDashboard
from tomea.core.harness import get_harness_code
from tomea.engine.executor import ModalExecutor
from tomea.agents.synthesizer import PaperSynthesizer
from tomea.templates.peft_template import PEFTTemplate
from tomea.analysis.analyzer import ResultAnalyzer, create_dataset_profile_from_data, experiment_result_from_metrics
from tomea.utils.artifacts import ArtifactManager
from tomea.utils.model_inspector import inspect_model 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_real_error(metrics: Dict) -> str:
    """
    Extract the actual error from metrics, parsing logs if needed.
    
    Priority:
    1. Full traceback from logs (best)
    2. Short error message (fallback)
    """
    logs = metrics.get('logs', '')
    short_error = metrics.get('error', 'Unknown Error')
    
    if 'Traceback' in logs:
        # Find the last traceback (most recent error)
        traceback_start = logs.rfind('Traceback (most recent call last):')
        if traceback_start != -1:
            # Get everything from the traceback onwards
            error_section = logs[traceback_start:]
            
            # Limit to 2000 chars to avoid overwhelming the LLM
            if len(error_section) > 2000:
                error_section = error_section[:1000] + "\n...[truncated]...\n" + error_section[-1000:]
            
            return error_section
    
    if short_error and short_error != 'Unknown Error':
        # If the error is suspiciously short (like "None"), try to find context in logs
        if len(short_error) < 20 and logs:
            lines = logs.split('\n')
            for i, line in enumerate(reversed(lines)):
                if 'Error' in line or 'Exception' in line or 'KeyError' in line:
                    # Get some context (5 lines before and after)
                    start_idx = max(0, len(lines) - i - 6)
                    end_idx = min(len(lines), len(lines) - i + 5)
                    context = '\n'.join(lines[start_idx:end_idx])
                    return context
        
        return short_error
    
    return 'Unknown Error - No traceback found in logs'


def backup_surgeon(code: str, error: str) -> str:
    """
    Manual string surgery for stubborn bugs the LLM can't fix.
    """
    lines = code.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if re.search(r'def\s+forward\s*\(', line) and "num_items_in_batch" in error:
            fixed_lines.append(line)
            indent = len(line) - len(line.lstrip()) + 4
            spaces = " " * indent
            fixed_lines.append(f"{spaces}# --- BACKUP SURGEON FIX ---")
            fixed_lines.append(f"{spaces}kwargs.pop('num_items_in_batch', None)")
            fixed_lines.append(f"{spaces}kwargs.pop('cache_position', None)")
            fixed_lines.append(f"{spaces}kwargs.pop('past_key_values', None)")
            continue
        
        if "chunk_size=" in line and "Boolean value of Tensor" in error:
            if "int(" not in line:  # Don't double-wrap
                fixed_line = re.sub(
                    r'chunk_size\s*=\s*([^,\)]+)',
                    r'chunk_size=int(\1)',
                    line
                )
                fixed_lines.append(fixed_line)
                continue
        
        if "super().__init__(config" in line and ("KeyError: None" in error or "_attn_implementation" in error):
            indent = len(line) - len(line.lstrip())
            spaces = " " * indent
            # Add the fix BEFORE super().__init__
            fixed_lines.append(f"{spaces}# --- BACKUP SURGEON: Fix _attn_implementation ---")
            fixed_lines.append(f"{spaces}if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:")
            fixed_lines.append(f"{spaces}    config._attn_implementation = 'eager'")
            fixed_lines.append(line)
            continue
        
        fixed_lines.append(line)
    
    result = "\n".join(fixed_lines)
    
    # Double-check: if we didn't make ANY changes, add a comment so we know
    if result == code:
        logger.warning("⚠️ Backup Surgeon couldn't identify a fix pattern for this error")
    
    return result


def robust_download(url: str, local_path: str):
    if os.path.exists(local_path):
        try:
            pd.read_csv(local_path, nrows=5)
            return 
        except Exception:
            os.remove(local_path)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")


def safe_load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')


async def fix_code_with_llm(llm: ChatOpenAI, code: str, error: str, previous_attempts: List[str] = []) -> str:
    """
    Surgical Healer V3: Enhanced diagnostics + Better error patterns
    """
    
    
    # --- ENHANCED DIAGNOSTICS ---
    hints = []
    
    # Diagnosis A: Kwargs Hygiene
    if "unexpected keyword" in error or "multiple values" in error or "num_items_in_batch" in error:
        hints.append(
            "CRITICAL FIX: The library is injecting metadata args that crash the model.\n"
            "SOLUTION: At the TOP of forward(), add:\n"
            "  kwargs.pop('num_items_in_batch', None)\n"
            "  kwargs.pop('cache_position', None)\n"
            "  kwargs.pop('past_key_values', None)"
        )

    # Diagnosis B: Mixed Precision
    if "types match" in error and "Float" in error:
        hints.append(
            "CRITICAL FIX: Cast tensor during assignment:\n"
            "  target[mask] = source.to(target.dtype)"
        )

    # Diagnosis C: Missing Entry Point
    if "has no attribute 'get_model'" in error:
        hints.append(
            "CRITICAL FIX: Restore the `get_model(base_model_name, num_labels)` function."
        )
    
    # Diagnosis D: Missing Helper Method
    if "object has no attribute" in error and "transpose_for_scores" in error:
        hints.append(
            "CRITICAL FIX: Define `transpose_for_scores` inside your class.\n"
            "Don't rely on inheritance - copy the method implementation."
        )
    
    # Diagnosis E: Boolean Tensor Error
    if "Boolean value of Tensor" in error:
        hints.append(
            "CRITICAL FIX: You passed a PyTorch Tensor where a Python int/bool is expected.\n"
            "Common culprits:\n"
            "  - chunk_size=config.chunk_size → chunk_size=int(config.chunk_size)\n"
            "  - if tensor > 0: → if int(tensor) > 0:\n"
            "SOLUTION: Wrap in int(...) or use .item() for scalar tensors."
        )
    
    # Diagnosis F: KeyError: None (NEW!)
    if "KeyError: None" in error or "_attn_implementation" in error:
        hints.append(
            "CRITICAL FIX: HuggingFace BertAttention expects `config._attn_implementation` but it's None.\n"
            "SOLUTION #1 (Recommended): Set it before calling super():\n"
            "  def __init__(self, config, position_embedding_type=None):\n"
            "      if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:\n"
            "          config._attn_implementation = 'eager'\n"
            "      super().__init__(config, position_embedding_type)\n"
            "\n"
            "SOLUTION #2: Don't inherit from BertAttention - use nn.Module instead."
        )
    
    # Diagnosis G: BERT_SELF_ATTENTION_CLASSES KeyError
    if "BERT_SELF_ATTENTION_CLASSES" in error:
        hints.append(
            "CRITICAL FIX: You're trying to use HuggingFace's internal attention class registry.\n"
            "SOLUTION: Either:\n"
            "  1. Set config._attn_implementation = 'eager' before super().__init__()\n"
            "  2. Import and use BertSelfAttention directly instead of using the registry"
        )

    hint_block = "\n".join([f"-> {h}" for h in hints]) if hints else "-> Analyze the error and fix the code."

    history_context = ""
    if previous_attempts:
        history_context = "\nPREVIOUS FAILED ATTEMPTS:\n" + "\n".join(previous_attempts)
        history_context += "\n\n⚠️ CRITICAL: The above errors happened AFTER your previous fixes. DO NOT try the same fix again."

   

    # --- THE PROMPT ---
    prompt = f"""
You are a Senior PyTorch Debugger fixing a crashed adapter.

ERROR (FULL TRACEBACK):
{error}

MANDATORY FIXES:
{hint_block}

BROKEN CODE:
{code}

TASK: 
1. Read the error carefully
2. Apply the mandatory fixes
3. Return the COMPLETE fixed code (not just snippets)

CRITICAL: Return ONLY the full Python code. No explanations, no markdown.
"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a Senior PyTorch Debugger. Return ONLY code."),
        HumanMessage(content=prompt)
    ])
    
    new_code = response.content.replace("```python", "").replace("```", "").strip()
    
    # Ensure we actually have imports at the start
    if "import torch" in new_code:
        new_code = new_code[new_code.find("import torch"):]

    # --- BACKUP SURGEON (Force-Fix) ---
    if new_code == code:
        logger.warning("🚑 Healer failed. Engaging Backup Surgeon...")
        new_code = backup_surgeon(new_code, error)

    return new_code


# =============================================================================
# MAIN ENGINE
# =============================================================================

async def run_comparison_engine(papers, dataset_path, llm_client, console) -> Dict:
    # 1. Setup
    artifacts = ArtifactManager()
    synthesizer = PaperSynthesizer(llm_client)
    peft_template = PEFTTemplate()
    executor = ModalExecutor()
    analyzer = ResultAnalyzer(llm_client=llm_client, baseline_method=papers[0]['name'])
    
    # 2. Data
    effective_data_path = dataset_path
    try:
        if dataset_path.startswith("http"):
            filename = dataset_path.split("/")[-1]
            if not filename.endswith(".csv"): filename = "temp.csv"
            os.makedirs("cache", exist_ok=True)
            effective_data_path = f"cache/{filename}"
            robust_download(dataset_path, effective_data_path)
        
        df = safe_load_dataset(effective_data_path)
        profile = create_dataset_profile_from_data(df.to_dict('records'))
    except Exception as e:
        console.print(f"[red]Data Error: {e}[/red]")
        traceback.print_exc()
        return {"success": False}

    # 3. Dashboard
    dashboard = ParallelDashboard(len(papers), [p['name'] for p in papers])
    
    # 4. Worker
    async def process_paper(index, paper):
        run_dir = artifacts.create_run(paper['name'], paper['type'])
        
        # --- LOGGING SETUP ---
        log_file = os.path.join(run_dir, "run.log")
        
        def write_log(msg):
            """Writes a message to the persistent log file."""
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")
        
        def update_status(msg):
            """Updates dashboard AND writes to log file."""
            dashboard.update_paper(index, msg)
            write_log(f"[STATUS] {msg}")

        write_log(f"=== STARTING RUN: {paper['name']} ===")
        write_log(f"Run Directory: {run_dir}")
        
        # --- PHASE 1: GENERATION ---
        dashboard.update_status(index, "Generating...")
        try:
            if paper['type'] == 'peft':
                code = peft_template.generate_adapter("", {"method_type": "LORA"}).code
            else:
                base_model_name = "bert-base-uncased"
                update_status("Inspecting Model...")
                specs = inspect_model(base_model_name)
                
                # Synthesis
                code = await asyncio.to_thread(
                    synthesizer.synthesize_from_arxiv, 
                    paper['arxiv'],
                    base_model=base_model_name,
                    model_specs=specs['str'],
                    status_callback=update_status 
                )
                write_log("--- CODE GENERATED SUCCESSFULLY ---")
                
        except Exception as e:
            error_msg = str(e)
            dashboard.mark_failed(index)
            update_status(f"Gen Failed: {error_msg[:20]}")
            write_log(f"!!! CRITICAL GENERATION ERROR: {error_msg}")
            traceback.print_exc()
            return None
        
        
        # --- PHASE 2: EXECUTION ---
        error_history = []
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES + 1):
            dashboard.update_status(index, f"Training (Attempt {attempt+1})...")
            write_log(f"\n=== TRAINING ATTEMPT {attempt+1} ===")
            
            artifacts.save_code(run_dir, f"adapter_v{attempt}.py", code)
            
            def log_cb(line): 
                dashboard.update_paper(index, line)
            
            try:
                metrics = await asyncio.to_thread(
                    executor.run_experiment,
                    harness_code=get_harness_code(),
                    adapter_code=code,
                    dataset_url=effective_data_path,
                    method_name=paper['name'],
                    log_callback=log_cb
                )
                
                # --- DUMP REMOTE LOGS ---
                remote_logs = metrics.get("logs", "")
                if remote_logs:
                    write_log("\n--- [START MODAL LOGS] ---")
                    write_log(remote_logs)
                    write_log("--- [END MODAL LOGS] ---\n")
                
                if metrics.get('status') == 'success':
                    acc = metrics.get('accuracy', 0)
                    dashboard.mark_complete(index, f"Acc: {acc:.2f}")
                    write_log(f"*** SUCCESS: Accuracy {acc:.2f} ***")
                    return experiment_result_from_metrics(metrics, paper['name'])
                
                else:
                    # --- EXTRACT THE ERROR ---
                    error_msg = extract_real_error(metrics)
                    write_log(f"!!! FAILURE: {error_msg}")
                    error_history.append(f"Attempt {attempt+1}: {error_msg[:200]}...")
                    
                    if attempt < MAX_RETRIES:
                        
                        dashboard.update_status(index, "[red] CRASH DETECTED! HEALING...  [/red]")
                        
                        
                        write_log("\n" + "="*40)
                        write_log("[bold red] CRASH DETECTED![/bold red]")
                        write_log(f"[yellow]{error_msg[:100]}...[/yellow]")
                        write_log("[bold green]🩹 INITIATING HEALER PROTOCOL...[/bold green]")
                        write_log("="*40 + "\n")
                        
                        dashboard.update_paper(index, "[bold red] CRASH DETECTED! [/bold red]")

                        
                        import time
                        time.sleep(4)
                        
                        
                        write_log("--- INITIATING HEALING ---")
                        
                        
                        new_code = await fix_code_with_llm(llm_client, code, error_msg, previous_attempts=error_history)
                        
                        if new_code == code:
                            write_log("!!! WARNING: Healer returned IDENTICAL code.")
                        else:
                            write_log("--- Healer modified the code ---")
                        
                        code = new_code
                    else:
                        dashboard.mark_failed(index)
                        update_status(f"Failed: {error_msg[:30]}")
                        return None
                        
            except Exception as e:
                console.print(f"\n[red]CRITICAL EXECUTION ERROR ({paper['name']}):[/red]")
                traceback.print_exc()
                write_log(f"CRITICAL EXECUTION EXCEPTION: {e}")
                return None

    # 5. Launch
    from tomea.engine.executor import app
    class DashWrapper:
        def __init__(self, d): self.d = d
        def __rich__(self): return self.d.render()

    with app.run():
        with Live(DashWrapper(dashboard), refresh_per_second=10, console=console):
            tasks = [process_paper(i, p) for i, p in enumerate(papers)]
            results = await asyncio.gather(*tasks)
            experiment_results = [r for r in results if r is not None]

    if experiment_results:
        return {
            "success": True, 
            "report": analyzer.analyze_and_recommend(experiment_results, profile, {}).full_report,
            "results": experiment_results
        }
    return {"success": False, "error": "All runs failed"}