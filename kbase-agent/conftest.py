import sys
from pathlib import Path

# 保证不依赖 editable 安装也能 import app.* / eval.*（例如直接跑 pytest）
sys.path.insert(0, str(Path(__file__).resolve().parent))
