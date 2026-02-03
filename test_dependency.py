import asyncio
import logging
from dotenv import load_dotenv
from tomea.core.orchestrator import TomeaOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Mamba requires specific custom kernels. 
# Let's see if the Synthesizer tries to import 'mamba_ssm' (which will fail)
# or if it tries to implement the SSM math using standard torch (which is what we want for a prototype).
MAMBA_ID = "2312.00752"
MAMBA_TEXT = """
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
We introduce a structured state space model (SSM) architecture...
"""

async def main():
    orchestrator = TomeaOrchestrator(mode="auto")
    print("--- 🧪 Testing Dependency Handling (Mamba) ---")
    
    # We expect SFTHarness (Architecture)
    await orchestrator.run_mission(arxiv_id=MAMBA_ID, paper_text=MAMBA_TEXT)

if __name__ == "__main__":
    asyncio.run(main())