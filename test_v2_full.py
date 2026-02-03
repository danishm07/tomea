import asyncio
import logging
from dotenv import load_dotenv
from tomea.core.orchestrator import TomeaOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Test DPO Paper
ID = "2305.18290" 
TEXT = """
CODE REPOSITORY: [https://github.com/eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)
Direct Preference Optimization (DPO).
"""

async def main():
    orchestrator = TomeaOrchestrator(mode="auto")
    
    # Passing None for dataset_name triggers the Data Engine's default logic
    # It should pick 'Anthropic/hh-rlhf' for DPO
    await orchestrator.run_mission(
        arxiv_id=ID, 
        paper_text=TEXT,
        dataset_name=None 
    )

if __name__ == "__main__":
    asyncio.run(main())