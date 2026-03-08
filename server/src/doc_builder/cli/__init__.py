"""
CLI module.
"""

from doc_builder.cli.setup import main as setup_main
from doc_builder.cli.status import main as status_main

__all__ = ["setup_main", "status_main"]
