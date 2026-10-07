from pathlib import Path
current_file = Path(__file__).resolve()
PROJECT_ROOT = Path.cwd()

TEMPLATES_PATH = PROJECT_ROOT / "app" / "frontend" / "templates"

DATA_PATH = PROJECT_ROOT / "app" / "backend" / "data.json"