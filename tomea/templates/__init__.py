"""Method templates for different paper types."""

from .base import BaseTemplate, AdapterCode
from .peft_template import PEFTTemplate

__all__ = [
    'BaseTemplate',
    'AdapterCode',
    'PEFTTemplate',
]