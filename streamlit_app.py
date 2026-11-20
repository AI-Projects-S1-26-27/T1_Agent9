"""Entry point for hosted Streamlit (for example Streamlit Community Cloud).

Owner: Rohit Gangil, Reviewer: Atharv Singhal

Put ``streamlit_app.py`` as the app file. It makes the ``src`` folder importable
and runs the real app in ``src/resource_war/demo/app.py``.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
runpy.run_path(str(HERE / "src" / "resource_war" / "demo" / "app.py"), run_name="__main__")
