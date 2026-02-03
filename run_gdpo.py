import asyncio
import os
import logging
from dotenv import load_dotenv
from tomea.core.orchestrator import TomeaOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# REAL PAPER: DPO (2305.18290)
# We Pre-pend the GitHub link to ensure detection.
REAL_ID = "2305.18290" 
REAL_TEXT = """
CODE REPOSITORY: https://github.com/eric-mitchell/direct-preference-optimization

Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
We propose DPO, a simple algorithm that implicitly optimizes the same objective as RLHF...
"""

async def main():
    orchestrator = TomeaOrchestrator(mode="auto")
    print(f"\n--- 🧪 Testing V2 Pipeline on REAL Paper (DPO) ---")
    
    await orchestrator.run_mission(
        arxiv_id=REAL_ID, 
        paper_text=REAL_TEXT
    )

if __name__ == "__main__":
    asyncio.run(main())