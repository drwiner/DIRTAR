"""
DIRTAR: Discovery of Inference Rules from Text for Action Recognition

A modernized implementation of the DIRT algorithm (Lin and Pantel, 2001)
with modifications for action recognition.
"""

# Import modern implementation by default
from .modern_dirtar import DIRTARProcessor, DependencyInfo, Entry, WordNetHelper

__version__ = "2.0.0"
__author__ = "David Winer (original), Modernized 2024"

__all__ = ["DIRTARProcessor", "DependencyInfo", "Entry", "WordNetHelper"]