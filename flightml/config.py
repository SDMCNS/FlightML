from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FLIGHTML_DATA_DIR", PROJECT_ROOT / "data"))
PARQUET_DIR = DATA_DIR / "parquet"
CACHE_DIR = PROJECT_ROOT / ".cache"
WEATHER_CACHE_DIR = CACHE_DIR / "weather"

# Ensure cache dirs exist
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

