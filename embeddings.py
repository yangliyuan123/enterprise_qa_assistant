"""豆包多模态 Embedding 封装 — 使用 /embeddings/multimodal 端点"""
from typing import List
import requests
from langchain_core.embeddings import Embeddings
from utils.helpers import logger
import config


class DoubaoEmbeddings(Embeddings):
    """豆包多模态 Embedding 模型，调用 /embeddings/multimodal 端点

    多模态端点要求 input 格式为 [{"type": "text", "text": "..."}]
    且响应为 {"data": {"embedding": [...]}}，不支持批量请求
    """

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_EMBEDDING_BASE_URL.rstrip("/")
        self.model = config.DOUBAO_EMBEDDING_MODEL

    def _embed_single(self, text: str) -> List[float]:
        """单条文本 embedding"""
        url = f"{self.base_url}/embeddings/multimodal"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": text}],
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["data"]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding（逐条请求）"""
        total = len(texts)
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self._embed_single(text)
                embeddings.append(emb)
                if total > 1 and (i == 0 or (i + 1) % 10 == 0 or i == total - 1):
                    logger.info(f"Embedding 进度: {i + 1}/{total}")
            except Exception as e:
                logger.error(f"Embedding 第 {i} 条失败: {e}")
                raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条 query embedding"""
        return self._embed_single(text)
