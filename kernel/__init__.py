"""
GAIA Kernel - The cognitive core of the General AI Architecture
"""

from .core import GAIAKernel
from .attention import AttentionManager
from .context import ContextManager

__all__ = ['GAIAKernel', 'AttentionManager', 'ContextManager']