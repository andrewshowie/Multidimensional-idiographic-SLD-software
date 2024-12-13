# tests/__init__.py
"""
Test suite for the Linguistic Analyzer package.
Includes integration and component tests.
"""

from .integration_test import test_full_integration, verify_components, run_system_test

__all__ = [
    'test_full_integration',
    'verify_components',
    'run_system_test'
]

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # seconds
    'test_messages': [
        "This is a simple test message.",
        "The quick brown fox jumps over the lazy dog.",
        "We are conducting an empirical analysis of the methodology.",
        "This contains sum misspelled wurds and erors.",
        "When analyzing complex theoretical frameworks, researchers must consider multiple variables."
    ],
    'delay_between_tests': 2  # seconds
}

# Test status codes
class TestStatus:
    SUCCESS = 0
    COMPONENT_FAILURE = 1
    INTEGRATION_FAILURE = 2
    TIMEOUT = 3

# Version info
__version__ = '1.0.0'