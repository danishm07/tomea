import asyncio
from tomea.core.orchestrator import TomeaOrchestrator

async def main():
    # Test Manual Override
    orch = TomeaOrchestrator(mode='sft')
    await orch.run_mission('arxiv:1234', 'This is a paper about Transformers')

if __name__ == '__main__':
    asyncio.run(main())
