"""
Tomea Interactive Menu - The 'Face' of the product.
"""
import logging
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.status import Status
from rich import box

from tomea.ui.theme import Theme
from tomea.utils.arxiv_parser import get_paper_data

console = Console()

# Standard Benchmarks
PRESETS = {
    "imdb": {"name": "IMDB Sentiment", "desc": "Binary sentiment (50k reviews)", "url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"},
    "sst2": {"name": "SST-2", "desc": "Stanford Sentiment Treebank", "url": "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"},
    "ag_news": {"name": "AG News", "desc": "News classification (4 classes)", "url": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"}
}

class MainMenu:
    """Handles all interactive user input with a pro aesthetic."""
    
    @staticmethod
    def select_dataset() -> str:
        """Show table of datasets and return URL/Path."""
        console.print(f"\n[bold]{Theme.BULLET} Step 1: Select Dataset[/bold]")
        console.print()
        
        # Create a nice looking table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim white")
        table.add_column("#", style="yellow", width=4, justify="center")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Name", style="white", width=20)
        table.add_column("Description", style="dim")

        # Add Presets
        i = 1
        keys = list(PRESETS.keys())
        for key in keys:
            p = PRESETS[key]
            table.add_row(str(i), key, p["name"], p["desc"])
            i += 1
            
        # Add Custom Options (Separated visually)
        table.add_section()
        table.add_row(str(i), "local", "Local File", "Upload a .csv from disk")
        table.add_row(str(i+1), "url", "Custom URL", "Download from web link")
        
        console.print(table)
        console.print()
        
        # Logic
        choices = [str(x) for x in range(1, i+2)]
        choice_idx = int(Prompt.ask("Choose option", choices=choices, default="1"))
        
        console.print() # Spacer
        
        if choice_idx <= len(keys):
            key = keys[choice_idx - 1]
            console.print(f"   {Theme.SUCCESS} Selected: [bold cyan]{PRESETS[key]['name']}[/bold cyan]")
            return PRESETS[key]["url"]
        elif choice_idx == len(keys) + 1:
            path = Prompt.ask("Enter path to CSV")
            return path
        else:
            url = Prompt.ask("Enter URL")
            return url

    @staticmethod
    def build_paper_list() -> List[Dict[str, str]]:
        """Guide user to build the list of papers to test."""
        console.print(f"\n[bold]{Theme.BULLET} Step 2: Define Candidates[/bold]")
        papers = []
        
        # 1. Ask for Baseline
        if Confirm.ask("Include standard baseline (LoRA)?", default=True):
            papers.append({"name": "LoRA", "type": "peft", "arxiv": "2106.09685"})
            console.print(f"   {Theme.SUCCESS} Added LoRA")

        # 2. Add Custom Papers loop
        while True:
            console.print()
            if not Confirm.ask("Add a novel paper?", default=len(papers) == 0):
                break
            
            console.print()
            arxiv_id = Prompt.ask("Enter ArXiv ID (e.g. 2505.06708)")
            
            # AUTO-FETCH MAGIC with Spinner
            paper_name = f"Paper {arxiv_id}"
            
            with Status(f"[bold cyan]{Theme.THINKING} Fetching metadata...[/bold cyan]", spinner="dots"):
                try:
                    data = get_paper_data(arxiv_id)
                    if data and data.get('title'):
                        fetched_title = data['title'].strip()
                        # Clean up title (remove newlines)
                        fetched_title = " ".join(fetched_title.split())
                except Exception as e:
                    fetched_title = None

            # Smart Confirmation
            final_name = paper_name
            if fetched_title:
                console.print(f"   Found: [italic]{fetched_title}[/italic]")
                if Confirm.ask("   Use this title?", default=True):
                    final_name = fetched_title
                else:
                    final_name = Prompt.ask("   Enter custom name")
            else:
                final_name = Prompt.ask("   Enter name", default=paper_name)
            
            papers.append({
                "name": final_name,
                "type": "novel",
                "arxiv": arxiv_id
            })
            console.print(f"   {Theme.SUCCESS} Added: [cyan]{final_name}[/cyan]")

        return papers

    @staticmethod
    def show_summary(papers, dataset):
        """Show final confirmation table."""
        console.print(f"\n[bold]{Theme.BULLET} Step 3: Execution Plan[/bold]")
        
        # Build Candidate List string
        candidate_list = ""
        for p in papers:
            p_type = "[green]PEFT[/green]" if p['type'] == 'peft' else "[magenta]Novel[/magenta]"
            candidate_list += f"{Theme.BULLET} {p['name']} ({p_type})\n"
        
        grid = Table.grid(padding=1)
        grid.add_column(style="bold white", justify="right", width=12)
        grid.add_column(style="dim white")
        
        # Truncate dataset URL for display if too long
        disp_dataset = dataset
        if len(dataset) > 60:
            disp_dataset = dataset[:30] + "..." + dataset[-25:]
            
        grid.add_row("Dataset:", disp_dataset)
        grid.add_row("Candidates:", str(len(papers)))
        grid.add_row("", candidate_list.strip()) # Empty label for list
        
        console.print(Panel(grid, title="Ready to Launch", border_style="cyan", box=box.ROUNDED, padding=(1, 2)))
        
        return Confirm.ask("\nLaunch Experiment?", default=True)