
from tomea.core.orchestrator import TomeaOrchestrator as Orchestrator
import asyncio
import dotenv
dotenv.load_dotenv()
async def test():
    orch = Orchestrator()
    result = await orch.run_mission(
        arxiv_id='2601.05242'  # GDPO paper
    )
    print('Success:', result)
asyncio.run(test())
