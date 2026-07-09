"""检索器优化 — 混合检索、重排序、查询重写、元数据过滤"""
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from utils.helpers import logger
import config


class OptimizedRetriever:
    """优化检索器：混合检索 + 重排序 + 查询重写 + 元数据过滤"""

    def __init__(self, vectorstore_manager):
        self.vs_manager = vectorstore_manager
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=config.DEEPSEEK_MODEL,
                openai_api_key=config.DEEPSEEK_API_KEY,
                openai_api_base=config.DEEPSEEK_BASE_URL,
                temperature=0.1,
            )
        return self._llm

    def build_ensemble_retriever(
        self, documents: list[Document], weights: list[float] | None = None
    ):
        """构建混合检索器（向量 + BM25）"""
        if weights is None:
            weights = [0.7, 0.3]

        vector_retriever = self.vs_manager.as_retriever(k=config.RETRIEVAL_K)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = config.RETRIEVAL_K

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights,
        )
        logger.info(f"混合检索器已构建 (向量:{weights[0]}, BM25:{weights[1]})")
        return ensemble

    def with_reranking(self, base_retriever):
        """对检索结果进行 LLM 重排序"""
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        return compression_retriever

    def rewrite_query(self, original_query: str, num_sub_queries: int = 3) -> list[str]:
        """将查询重写为多个子查询"""
        prompt = f"""将以下查询重写为{num_sub_queries}个更具体的子查询，每个子查询应该从不同角度探索原始问题：

原始查询：{original_query}

请按以下格式输出（每行一个子查询，不要编号）：
[子查询1]
[子查询2]
[子查询3]"""

        response = self.llm.invoke(prompt)
        lines = response.content.strip().split("\n")
        sub_queries = [line.strip("- 1234567890. ") for line in lines if line.strip()]
        sub_queries = [q for q in sub_queries if q]
        logger.info(f"查询重写: '{original_query[:30]}...' → {len(sub_queries)} 个子查询")
        return sub_queries or [original_query]

    def filter_by_metadata(
        self, docs: list[Document], filters: dict
    ) -> list[Document]:
        """按元数据过滤"""
        if not filters:
            return docs
        filtered = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered
