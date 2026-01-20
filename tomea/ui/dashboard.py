import time
import random
import re
import math
from collections import deque

# Rich Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console

# The Retro Plotter
import asciichartpy as ac


class LossPlot:
    """
    Renders a Retro Ascii Chart at 75% width.
    Theme: Forest Green & Silver.
    """
    def __init__(self, steps, losses):
        self.steps = list(steps)
        self.losses = list(losses)

    def process_data(self, data, target_width):
        """
        Smart Resizing:
        1. If data is small, stretch it to fill the width (Elongate).
        2. If data is large, Resample/Scroll so updates are visible.
        """
        if not data: 
            return []
        
        # If we have very few points, stretch them nicely so it's not a "worm"
        if len(data) < target_width:
            # Linear Interpolation (Stretch)
            src_indices = [i for i in range(len(data))]
            scale = (len(data) - 1) / (target_width - 1) if target_width > 1 else 0
            target_indices = [i * scale for i in range(target_width)]
            
            new_data = []
            for x in target_indices:
                lower = int(math.floor(x))
                upper = int(math.ceil(x))
                weight = x - lower
                val = data[lower] * (1 - weight) + data[upper] * weight
                new_data.append(val)
            return new_data
            
        # If we have MORE data than width, we resample (compress) 
        # but we prioritize the latest data to keep it feeling "live"
        chunk_size = len(data) / target_width
        new_data = []
        for i in range(target_width):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            slice_data = data[start:max(start + 1, end)]
            if slice_data:
                new_data.append(sum(slice_data) / len(slice_data))
        return new_data

    def __rich_console__(self, console, options):
        # 1. Geometry Calculations
        # Calculate full available width then take 75% as requested
        full_width = max(20, options.max_width - 12)
        width = int(full_width * 0.75)  # <--- THE 75% CONDENSE FACTOR
        
        height = max(5, options.max_height - 4) 

        if not self.losses:
            yield Text("Waiting for data...", style="dim white")
            return

        # 2. Process Data (Stretch or Compress)
        chart_data = self.process_data(self.losses, width)
        
        # 3. Configure Retro Plot (Forest Green Theme)
        cfg = {
            "height": height,
            "format": "{:8.4f}", 
            "colors": [ac.green] 
        }
        
        # 4. Generate the Chart
        try:
            chart_str = ac.plot(chart_data, cfg)
        except Exception as e:
            yield Text(f"Error: {e}", style="red")
            return

        # 5. Render Components
        
        # Component A: Y-Axis Header
        yield Text("  (Loss) ^", style="dim white")

        # Component B: The Chart
        yield Text.from_ansi(chart_str)
        
        # Component C: X-Axis Numbers
        # We construct the labels based on the REAL step counts
        start_step = self.steps[0]
        end_step = self.steps[-1]
        mid_step = int((start_step + end_step) / 2)
        
        lbl_start = str(start_step)
        lbl_mid = str(mid_step)
        lbl_end = str(end_step)
        
        # Padding logic to align with chart area
        padding_width = 10 
        y_axis_spacer = " " * padding_width
        
        # Use the condensed 'width' for spacing calculations
        space_1 = (width // 2) - len(lbl_start) - (len(lbl_mid) // 2)
        space_2 = width - (width // 2) - (len(lbl_mid) // 2) - len(lbl_end)
        
        x_axis_nums = (
            f"{y_axis_spacer}"
            f"[bold white]{lbl_start}[/bold white]"
            f"{' ' * max(0, space_1)}"
            f"[dim white]{lbl_mid}[/dim white]"
            f"{' ' * max(0, space_2)}"
            f"[bold white]{lbl_end}[/bold white]"
        )
        yield Text.from_markup(x_axis_nums)

        # Component D: X-Axis Title
        label = "Steps →"
        # Center relative to the condensed width
        center_pad = padding_width + (width // 2) - (len(label) // 2)
        yield Text(" " * center_pad + label, style="dim white")


class TrainingDashboard:
    def __init__(self, paper_name: str = "Unknown"):
        self.paper_name = paper_name
        self.losses = deque(maxlen=2000)
        self.eval_losses = deque(maxlen=200)
        self.steps = deque(maxlen=2000)
        self.eval_steps = deque(maxlen=200)
        
        self.current_step = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.current_eval_loss = 0.0
        self.current_lr = 0.0
        self.total_steps = 1000 
        self.all_stdout = []
        
    def parse_line(self, line: str):
        self.all_stdout.append(line)
        
        if "'loss':" in line:
            loss_match = re.search(r"'loss': ([\d.]+)", line)
            lr_match = re.search(r"'learning_rate': ([\d.e-]+)", line)
            epoch_match = re.search(r"'epoch': ([\d.]+)", line)
            
            if loss_match:
                self.current_loss = float(loss_match.group(1))
                self.losses.append(self.current_loss)
                self.steps.append(self.current_step)
                
                if self.current_step > self.total_steps:
                    self.total_steps = self.current_step + 1000 
                
                self.current_step += 10 
                
            if lr_match: self.current_lr = float(lr_match.group(1))
            if epoch_match: self.current_epoch = float(epoch_match.group(1))
        
        if "'eval_loss':" in line:
            eval_match = re.search(r"'eval_loss': ([\d.]+)", line)
            if eval_match:
                self.current_eval_loss = float(eval_match.group(1))
                self.eval_losses.append(self.current_eval_loss)
                self.eval_steps.append(self.current_step)
    
    def render(self) -> Layout:
        layout = Layout()
        
        # Define Layout Grid
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=3),
            Layout(name="footer", size=10)
        )
        layout["body"].split_row(
            Layout(name="loss_plot", ratio=3),
            Layout(name="metrics", ratio=1)
        )
        
        # --- 1. HEADER ---
        display_name = (self.paper_name[:50] + '..') if len(self.paper_name) > 50 else self.paper_name
        
        header_text = (
            f"[bold green]{display_name}[/bold green]"
        )
        layout["header"].update(Panel(header_text, border_style="green"))
        
        # --- 2. PLOT ---
        loss_plot = LossPlot(self.steps, self.losses)
        layout["loss_plot"].update(
            Panel(loss_plot, border_style="green", title="[bold white]Training Loss[/bold white]")
        )
        
        # --- 3. METRICS ---
        metrics = Table.grid(padding=1)
        metrics.add_column(style="green bold", justify="right")
        metrics.add_column(style="white")
        
        metrics.add_row("Train Loss", f"{self.current_loss:.4f}")
        metrics.add_row("Eval Loss", f"{self.current_eval_loss:.4f}")
        metrics.add_row("LR", f"{self.current_lr:.2e}")
        metrics.add_row("Epoch", f"{self.current_epoch:.2f}")
        metrics.add_row("Step", f"{self.current_step} / {self.total_steps}")
        
        layout["metrics"].update(
            Panel(metrics, title="[green]Metrics[/green]", border_style="dim white")
        )
        
        # --- 4. FOOTER ---
        logs = "\n".join(self.all_stdout[-8:]) if self.all_stdout else ""
        layout["footer"].update(
            Panel(logs, title="[dim]Real-time Logs[/dim]", border_style="dim white")
        )
        
        return layout


if __name__ == "__main__":
    dashboard = TrainingDashboard(paper_name="Condensed 75% Graph")
    
    with Live(dashboard.render(), refresh_per_second=4, screen=True) as live:
        step = 0
        try:
            while True:
                # Add noise
                noise = (random.random() - 0.5) * 0.1
                loss_val = max(0.1, 2.5 * (0.999 ** step) + noise)
                lr_val = 5e-5 * (0.9995 ** step)
                
                log = f"{{'loss': {loss_val:.4f}, 'learning_rate': {lr_val:.2e}, 'epoch': {step/500:.2f}}}"
                dashboard.parse_line(log)
                
                if step > 0 and step % 100 == 0:
                    eval_val = loss_val * 1.05
                    dashboard.parse_line(f"{{'eval_loss': {eval_val:.4f}}}")

                live.update(dashboard.render())
                step += 2 # Faster step to see updates better
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            pass