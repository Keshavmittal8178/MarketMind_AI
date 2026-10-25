"""
app.py
------
Root entry point for Streamlit Community Cloud and local execution.
Redirects to and executes the standardized MarketMind AI dashboard in dashboard/app.py.
"""

from __future__ import annotations
import os
import sys

# Ensure root directory is always on python sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Execute the main dashboard
dashboard_file = os.path.join(ROOT_DIR, "dashboard", "app.py")
with open(dashboard_file, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, dashboard_file, "exec"), globals())
