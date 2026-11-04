"""统一读取 .env：网关地址 / key / 模型名。"""
import os, pathlib
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY  = os.getenv("LLM_API_KEY")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "deepseek-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding")
