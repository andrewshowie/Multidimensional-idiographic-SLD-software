# linguistic_analyzer/__init__.py
"""
Linguistic Analyzer package.
A real-time linguistic analysis suite for text processing.
"""

from .integrated_analyzer import IntegratedAnalyzer
from .visualizer import LinguisticVisualizer
from .chat_client import ChatClient
from .chat_server import ChatServer

__version__ = '1.0.0'

__all__ = [
    'IntegratedAnalyzer',
    'LinguisticVisualizer',
    'ChatClient',
    'ChatServer'
]
