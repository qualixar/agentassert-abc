"""Pytest configuration for dashboard tests.

Adds dashboard/ to sys.path so export_dashboard_data can be imported.
Run from agentassert-abc repo root:
    .venv/bin/python -m pytest dashboard/tests/ -v
"""
import sys
from pathlib import Path

# dashboard/ directory — where export_dashboard_data.py lives
sys.path.insert(0, str(Path(__file__).parent.parent))
