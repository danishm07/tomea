import asyncio
import os
import logging
from datetime import datetime

from tomea.core.router import ResearchRouter
from tomea.core.synthesizer import Synthesizer
from tomea.core.healer import Healer
from tomea.core.data_engine import DataEngine
from tomea.core.verifier import MathVerifier
from tomea.utils.llm import LLMClient
from tomea.core.modal_bridge import ModalExecutor

class TomeaOrchestrator:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.llm = LLMClient()
        self.synthesizer = Synthesizer(self.llm)
        self.healer = Healer(self.llm)
        self.data_engine = DataEngine()
        self.verifier = MathVerifier(self.llm)
        self.executor = ModalExecutor()

    async def run_mission(self, arxiv_id: str, paper_text: str, dataset_name: str = None):
        # 1. Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.abspath(f"runs/{arxiv_id}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"--- 🚀 Starting Mission: {arxiv_id} ---")
        
        # 1b. Fetch paper text BEFORE routing — router needs it to keyword-match.
        # If paper_text is empty or too short, pull the PDF now.
        if len(paper_text) < 500 and arxiv_id:
            try:
                from tomea.utils.arxiv_parser import get_paper_data
                print(f"--- 📄 Fetching paper text for {arxiv_id}... ---")
                data = get_paper_data(arxiv_id)
                if data:
                    paper_text = data["text"][:30000]
                    print(f"   ✅ Fetched {len(paper_text)} chars")
            except Exception as e:
                print(f"   ⚠️  PDF fetch failed: {e}. Router will use what's available.")
        
        # 2. Harness & Data (router now has actual paper text)
        HarnessClass = ResearchRouter.get_harness(self.mode, paper_text)
        harness = HarnessClass(paper_context={"id": arxiv_id})
        
        # Use the harness's declared mode — no more string matching on class names
        engine_mode = harness.mode
        print(f"--- 📊 Analysing Data (Mode: {engine_mode})... ---")
        
        data_context = self.data_engine.load_and_inspect(dataset_name, mode=engine_mode)
        if "error" in data_context:
            print(f"❌ Data Error: {data_context['error']}")
            return False

        # 3. Synthesize (now returns code + extraction metadata)
        print("--- 📝 Synthesizing Implementation... ---")
        code, extraction_metadata = await self.synthesizer.generate_implementation(
            harness, paper_text, arxiv_id, data_context
        )
        
        # Log extraction metadata for visibility
        self._save_file(run_dir, "extraction_analysis.json", 
                       __import__('json').dumps(extraction_metadata, indent=2))
        print(f"--- 📊 Extraction Confidence: {extraction_metadata.get('confidence', 'N/A')} ---")

        if extraction_metadata.get("confidence", 0) < 0.4:
            print(f"\n--- ⚠️  Insufficient Specification (Confidence: {extraction_metadata.get('confidence')}) ---")
            print(f"   Novel Contribution: {extraction_metadata.get('novel_contribution', 'Unknown')}")
            print(f"   Missing: Loss function, training algorithm, or pseudocode")
            print(f"   Cannot generate implementation without algorithmic specification.")
            print(f"   Suggestion: Find published paper or manually specify algorithm.")
            return False
        
        if extraction_metadata.get("smoke_test_feasible") in ("partial", "no"):
            print(f"   ⚠️  Smoke Test Note: {extraction_metadata.get('smoke_test_feasible')}")
        
        self._save_file(run_dir, "experiment.py", code)
        
        # 4. Verify
        print("--- 🧐 Running Math Verification... ---")
        await self.verifier.verify_logic(code, run_dir)
        
        # 5. Execute with heal loop
        max_retries = 3
        success = False
        
        # Track full attempt history for healer context
        attempt_history = []
        
        for attempt in range(max_retries + 1):
            print(f"\n--- ⚡ Execution Attempt {attempt + 1} ---")
            
            run_success, full_logs = await self.executor.run(code)
            self._save_file(run_dir, f"run_logs_v{attempt}.txt", full_logs)
            
            # Short-circuit: Modal infrastructure errors (image build failures,
            # platform outages, etc.) cannot be fixed by patching experiment.py.
            # Break immediately — don't waste LLM calls on the healer.
            if "MODAL SYSTEM ERROR" in full_logs:
                print("\n--- 🛑 Modal Infrastructure Failure — not healable ---")
                print(f"      {full_logs.strip()}")
                print("--- 💀 Mission Failed: platform error. Check Modal dashboard. ---")
                break
            
            # Record this attempt for healer history
            attempt_history.append({
                "attempt": attempt + 1,
                "code": code,
                "logs": full_logs,
                "exit_success": run_success
            })
            
            if run_success:
                # Modal said it exited cleanly — but did it actually produce valid output?
                # This is where validate_logs catches silent failures (NaN, no loss output, etc.)
                logs_valid = harness.validate_logs(full_logs)
                
                if logs_valid:
                    print("\n--- ✅ Mission Success! ---")
                    success = True
                    break
                else:
                    print("\n--- ⚠️  Exit code clean but output invalid (silent failure) ---")
                    run_success = False  # Treat as failure, fall through to heal
                    if attempt < max_retries:
                        print("--- 🚑 Healer analyzing silent failure... ---")
                        code = await self.healer.heal_code(code, full_logs, harness, attempt_history)
                        self._save_file(run_dir, f"experiment_v{attempt+1}.py", code)
                        print(f"--- 🩹 Code healed. Retrying... ---")
                    else:
                        print("--- 💀 Mission Failed: output never validated after max retries. ---")
            else:
                print(f"\n--- ❌ Crash Detected (Attempt {attempt + 1}) ---")
                if attempt < max_retries:
                    print("--- 🚑 Healer analyzing crash... ---")
                    code = await self.healer.heal_code(code, full_logs, harness, attempt_history)
                    self._save_file(run_dir, f"experiment_v{attempt+1}.py", code)
                    print(f"--- 🩹 Code healed. Retrying... ---")
                else:
                    print("--- 💀 Mission Failed after max retries. ---")

        print("\n" + "="*50)
        print(f"🏁 MISSION COMPLETE: {'SUCCESS' if success else 'FAILURE'}")
        print(f"📂 Artifacts: {run_dir}")
        print("="*50)
        
        return success

    def _save_file(self, folder, filename, content):
        path = os.path.join(folder, filename)
        with open(path, "w") as f: f.write(content)