import asyncio
import logging
from dotenv import load_dotenv
from tomea.core.orchestrator import TomeaOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

TEST_CASES = [
    {
        "name": "ORPO (Monolithic)",
        "id": "2403.07691",
        "desc": "ORPO: Monolithic Preference Optimization without Reference Model. ORPO adds an odds-ratio penalty to the NLL loss, combining SFT and Alignment."
    },
    {
        "name": "SPIN (Self-Play)",
        "id": "2401.01335",
        "desc": "Self-Play Fine-Tuning Converts Weak Language Models to Strong Agents. SPIN uses an iterative SFT loop where the model plays against its previous version."
    },
    {
        "name": "WARP (Alignment)",
        "id": "2406.16768",
        "desc": "WARP: On the Alignment of Large Language Models via Word-Level Rewards. It modifies the KL divergence constraint."
    }
]

async def main():
    orchestrator = TomeaOrchestrator(mode="auto")
    print("="*60)
    print("🔬 TOMEA V2 STRESS TEST: ROUND 2")
    print("="*60)

    for test in TEST_CASES:
        print(f"\n\n>>> 🧪 TEST SUBJECT: {test['name']}")
        await orchestrator.run_mission(
            arxiv_id=test['id'], 
            paper_text=test['desc']
        )

if __name__ == "__main__":
    asyncio.run(main())