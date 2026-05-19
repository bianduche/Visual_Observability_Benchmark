"""
Visual Observability Benchmark - Core Module
=============================================
Provides fundamental image degradation and observability mathematical operations.

Author: Visual Observability Benchmark Team
"""

from .degradation import DegradationEngine
from .observability_math import generate_observability_map

__all__ = ['DegradationEngine', 'generate_observability_map']
