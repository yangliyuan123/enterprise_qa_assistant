"""
企业知识问答助手 — 统一配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


# ============================================
# LLM 模型配置
# ============================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_EMBEDDING_BASE_URL = os.getenv("DOUBAO_EMBEDDING_BASE_URL")
DOUBAO_EMBEDDING_MODEL = os.getenv("DOUBAO_EMBEDDING_MODEL", "doubao-embedding-vision-251215")

# ============================================
# 向量数据库配置
# ============================================
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
# 嵌入式 Chroma 持久化路径（Docker 不可用时的备选方案）
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")

# ============================================
# Elasticsearch 配置
# ============================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")

# ============================================
# 文档处理参数
# ============================================
CHUNK_SIZE = 500          # 文档分割块大小
CHUNK_OVERLAP = 50        # 分割块之间的重叠
RETRIEVAL_K = 4           # 检索返回的文档数

# ============================================
# 文件路径
# ============================================
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
