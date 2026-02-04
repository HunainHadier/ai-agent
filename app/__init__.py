from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
