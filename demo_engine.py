import asyncio
import os
import sys
import re
from rich.console import Console
from rich.panel import Panel
from langchain_openai import ChatOpenAI

import tomea.core.harness
from tomea.core.harness import get_harness_code as original_get_harness
from tomea.ui.menu import MainMenu
from tomea.core.engine import run_comparison_engine
from dotenv import load_dotenv
load_dotenv()
console = Console()

# 1. SHORTEN TRAINING (For Demo Speed Only)

#def get_demo_harness_code():
#    code = original_get_harness()
#    pattern = r"TrainingArguments\s*\("
#    demo_args = """TrainingArguments(
#    max_steps=20, logging_steps=1, evaluation_strategy='steps', eval_steps=5, 
#    save_strategy='no', report_to='none',
#    """
#   if re.search(pattern, code):
#        return re.sub(pattern, demo_args, code, count=1)
#    return code

#tomea.core.harness.get_harness_code = get_demo_harness_code



# 2. MAIN RUNNER
async def main():
    console.print(Panel("[bold green]✨ DEMO[/bold green]\nSelf-Healing Enabled. Real Code Generation.", border_style="green"))

    dataset_url = MainMenu.select_dataset()
    papers = MainMenu.build_paper_list()
    
    if not MainMenu.show_summary(papers, dataset_url): return

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_client = ChatOpenAI(
        model="google/gemini-2.5-flash", 
        temperature=0.0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # RUN THE NEW SELF-HEALING ENGINE
    results = await run_comparison_engine(papers, dataset_url, llm_client, console)

    if results.get("success"):
        console.print("\n[bold green]✓ Success[/bold green]")
        console.print(Panel(results["report"]))
    else:
        console.print(f"\n[red]Failed: {results.get('error')}[/red]")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass