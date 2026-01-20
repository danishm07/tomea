import math
from collections import deque
from typing import List
import re

# Rich Imports
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console

# Retro Plotter
import asciichartpy as ac


class CompactLossPlot:
    """
    Compact loss plot with improved Y-axis height
    and a properly scaled X-axis with ticks.
    """

    def __init__(self, steps, losses, color=ac.green):
        self.steps = list(steps) if steps else []
        self.losses = list(losses) if losses else []
        self.color = color

    def process_data(self, data, target_width):
        """Resamples data to fit the target width exactly."""
        if not data:
            return []

        # If we have fewer points than width, linear interpolation (stretch)
        if len(data) < target_width:
            scale = (len(data) - 1) / (target_width - 1) if target_width > 1 else 0
            stretched = []
            for i in range(target_width):
                x = i * scale
                lo = int(math.floor(x))
                hi = int(math.ceil(x))
                w = x - lo
                # Safe interpolation
                val = data[lo] * (1 - w) + data[min(hi, len(data) - 1)] * w
                stretched.append(val)
            return stretched

        # If we have more points than width, average pooling (compress)
        chunk = len(data) / target_width
        compressed = []
        for i in range(target_width):
            start = int(i * chunk)
            end = int((i + 1) * chunk)
            # Handle edge case where slice is empty
            slice_data = data[start:max(start + 1, end)]
            if slice_data:
                compressed.append(sum(slice_data) / len(slice_data))
            else:
                # Fallback to previous value or 0 if empty
                compressed.append(compressed[-1] if compressed else 0)
        return compressed

    def __rich_console__(self, console, options):
        # 1. Calculate dimensions carefully
        # We subtract padding to ensure the ASCII art doesn't wrap
        width = max(10, options.max_width - 4) 
        height = max(5, options.max_height - 4) 

        if not self.losses:
            yield Text("Waiting for metrics...", style="dim white")
            return

        # 2. Resample data to fit width
        chart_data = self.process_data(self.losses, width)

        # 3. Configure asciichartpy
        cfg = {
            "height": height,
            "format": "{:6.3f}",
            "colors": [self.color],
        }

        try:
            chart_str = ac.plot(chart_data, cfg)
        except Exception as e:
            yield Text(f"Plot error: {e}", style="red")
            return

        # 4. CRITICAL FIX: no_wrap=True
        # This prevents Rich from reflowing the ASCII art and breaking the lines
        yield Text.from_ansi(chart_str, no_wrap=True, overflow="crop")

        # 5. X-Axis (Steps)
        if self.steps:
            start_step = self.steps[0]
            end_step = self.steps[-1]
            
            # Create a simple axis line
            axis_line = "─" * width
            
            # Create labels string (spaced out)
            label_str = f"{int(start_step)}"
            mid_val = int((start_step + end_step) / 2)
            end_val = int(end_step)
            
            mid_str = str(mid_val)
            end_str = str(end_val)
            
            # Calculate padding
            # Left align start, center align mid, right align end
            remaining = width - len(label_str) - len(end_str)
            if remaining > len(mid_str) + 2:
                pad_left = (remaining // 2) - len(mid_str)
                pad_right = remaining - pad_left - len(mid_str)
                labels = f"{label_str}{' ' * pad_left}{mid_str}{' ' * pad_right}{end_str}"
            else:
                # If too small, just show start/end
                pad = width - len(label_str) - len(end_str)
                labels = f"{label_str}{' ' * max(1, pad)}{end_str}"

            yield Text(axis_line, style="dim white", no_wrap=True)
            yield Text(labels, style="dim white", no_wrap=True)


class PaperTracker:
    """Tracks one training job"""

    def __init__(self, name: str, color=ac.green):
        self.name = name
        self.losses = deque(maxlen=400)
        self.eval_losses = deque(maxlen=100)
        self.steps = deque(maxlen=400)
        
        self.current_step = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.current_eval_loss = 0.0
        self.current_lr = 0.0
        self.total_steps = 1000

        self.status = "Initializing..."
        self.is_complete = False
        self.is_failed = False
        self.is_running = False
        self.color = color

        self.all_logs = []

    def parse_line(self, line: str):
        self.all_logs.append(line)

        # Basic parsing logic
        if "'loss':" in line:
            self.is_running = True
            loss_match = re.search(r"'loss':\s*([\d.]+)", line)
            lr_match = re.search(r"'learning_rate':\s*([\d.e-]+)", line)
            epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)

            if loss_match:
                val = float(loss_match.group(1))
                self.current_loss = val
                self.losses.append(val)
                self.steps.append(self.current_step)
                self.current_step += 1  # Increment step counter

            if lr_match:
                self.current_lr = float(lr_match.group(1))
            if epoch_match:
                self.current_epoch = float(epoch_match.group(1))

        if "'eval_loss':" in line:
            ev = re.search(r"'eval_loss':\s*([\d.]+)", line)
            if ev:
                self.current_eval_loss = float(ev.group(1))
                self.eval_losses.append(self.current_eval_loss)

    def render_compact(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(name="upper", ratio=3),
            Layout(name="logs", size=3)
        )

        layout["upper"].split_row(
            Layout(name="graph", ratio=3),
            Layout(name="metrics", ratio=1),
        )

        loss_plot = CompactLossPlot(self.steps, self.losses, self.color)
        layout["upper"]["graph"].update(
            Panel(loss_plot, title="[bold white]Loss Curve[/bold white]", border_style="dim white")
        )

        metrics = Table.grid(padding=1)
        metrics.add_column(justify="right", style="green bold")
        metrics.add_column(style="white")

        metrics.add_row("Loss:", f"{self.current_loss:.4f}")
        metrics.add_row("Eval:", f"{self.current_eval_loss:.4f}")
        metrics.add_row("LR:", f"{self.current_lr:.2e}")
        metrics.add_row("Epoch:", f"{self.current_epoch:.1f}")

        layout["upper"]["metrics"].update(
            Panel(metrics, title="Stats", border_style="dim white")
        )

        # Show last log line
        last_log = self.all_logs[-1] if self.all_logs else "Waiting..."
        # Clean up log line if it's too long
        if len(last_log) > 100: 
            last_log = last_log[:97] + "..."
            
        layout["logs"].update(
            Panel(
                Text(last_log, style="dim grey", no_wrap=True),
                title="Latest Log",
                border_style="dim black"
            )
        )

        if self.is_complete:
            icon = "✓"
            style = "green"
        elif self.is_failed:
            icon = "✗"
            style = "red"
        elif self.is_running:
            icon = "⚡"
            style = "blue"
        else:
            icon = "⋯"
            style = "white"

        title = f"[{style}]{icon} {self.name} | {self.status}[/{style}]"
        return Panel(layout, title=title, border_style=style)


class ParallelDashboard:
    def __init__(self, num_papers: int, paper_names: List[str]):
        # Removed cyan/blue which look bad on dark terminals sometimes
        colors = [ac.green, ac.magenta, ac.yellow, ac.red, ac.lightcyan]
        self.papers = [
            PaperTracker(name, colors[i % len(colors)])
            for i, name in enumerate(paper_names)
        ]

    def update_paper(self, index: int, line: str):
        if 0 <= index < len(self.papers):
            self.papers[index].parse_line(line)

    def update_status(self, index: int, status: str):
        if 0 <= index < len(self.papers):
            self.papers[index].status = status

    def mark_complete(self, index: int, status: str):
        if 0 <= index < len(self.papers):
            self.papers[index].is_complete = True
            self.papers[index].is_running = False
            self.papers[index].status = f"Complete ({status})"

    def mark_failed(self, index: int):
        if 0 <= index < len(self.papers):
            self.papers[index].is_failed = True
            self.papers[index].is_running = False
            self.papers[index].status = "Failed"

    def render(self) -> Layout:
        layout = Layout()
        
        # Header
        sections = [Layout(name="header", size=3)]
        
        # Create a grid for papers
        # If we have many papers, we might want to split into rows?
        # For now, let's keep vertical stack but make them smaller
        paper_sections = [Layout(name=f"paper{i}", ratio=1) for i in range(len(self.papers))]
        sections.extend(paper_sections)
        
        layout.split_column(*sections)

        complete = sum(p.is_complete for p in self.papers)
        running = sum(p.is_running for p in self.papers)
        failed = sum(p.is_failed for p in self.papers)

        header_text = (
            f" [bold]TOMEA ENGINE[/bold]  "
            f"[green]✓ {complete}[/green]  "
            f"[blue]⚡ {running}[/blue]  "
            f"[red]✗ {failed}[/red]"
        )
        layout["header"].update(Panel(header_text, style="blue"))

        for i, p in enumerate(self.papers):
            layout[f"paper{i}"].update(p.render_compact())

        return layout