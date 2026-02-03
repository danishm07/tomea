import asyncio
import logging
from dotenv import load_dotenv
from tomea.core.orchestrator import TomeaOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

TEST_CASES = [
    {
        "name": "LoRA (Architecture)",
        "id": "2106.09685",
        "desc": "Low-Rank Adaptation of Large Language Models. We freeze the pre-trained model weights and inject trainable rank decomposition matrices."
    },
    {
        "name": "DeepSeekMath (GRPO)",
        "id": "2402.03300",
        "desc": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning. We introduce Group Relative Policy Optimization (GRPO) which eliminates the critic model."
    },
    {
        "name": "SimPO (Reference-Free)",
        "id": "2405.14734",
        "desc": "SimPO: Simple Preference Optimization with a Reference-Free Reward. SimPO aligns models without a reference model by using the average log probability of the generated sequence."
    }
]

async def main():
    orchestrator = TomeaOrchestrator(mode="auto")
    
    print("="*60)
    print("🔬 TOMEA V2 STRESS TEST PROTOCOL")
    print("="*60)

    for test in TEST_CASES:
        print(f"\n\n>>> 🧪 TEST SUBJECT: {test['name']}")
        print(f">>> 📄 ArXiv ID: {test['id']}")
        
        # We pass the description as a fallback in case PDF fetch fails
        # But since these are real IDs, it SHOULD fetch the full PDF.
        await orchestrator.run_mission(
            arxiv_id=test['id'], 
            paper_text=test['desc'] # Fallback context
        )
        
    print("\n\n" + "="*60)
    print("✅ STRESS TEST COMPLETE. Check runs/ folder.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())