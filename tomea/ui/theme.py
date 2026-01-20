"""
Tomea UI Theme - Professional Unicode Aesthetics.
"""
from rich.style import Style

class Theme:
    """
    Unicode symbols and styles for a professional CLI look.
    Reference: Nerd Fonts / Unicode Block Elements
    """
    # Symbols
    LOGO = "◈"
    SUCCESS = "✓"
    FAILURE = "×"
    WARNING = "!"
    INFO = "ℹ"
    
    # Process indicators
    RUNNING = "›"
    WAITING = "·"
    THINKING = "⚡"  # or "⟳"
    
    # Structure
    ARROW = "→"
    BULLET = "•"
    SEPARATOR = "│"
    BLOCK = "█"
    
    # Colors/Styles
    STYLE_SUCCESS = Style(color="green", bold=True)
    STYLE_FAILURE = Style(color="red", bold=True)
    STYLE_WARNING = Style(color="yellow")
    STYLE_DIM = Style(color="white", dim=True)
    STYLE_HIGHLIGHT = Style(color="cyan", bold=True)
    
    # Borders (Box Drawing)
    BORDER_HEAVY = "heavy"
    BORDER_ROUNDED = "rounded"