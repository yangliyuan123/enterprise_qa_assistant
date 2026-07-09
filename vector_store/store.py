"""向量数据库管理 — 支持 Docker ChromaDB 服务端和本地嵌入式两种模式"""
import chromadb
from langchain_chroma import Chroma
from embeddings import DoubaoEmbeddings
from langchain_core.documents import Document
from utils.helpers import logger
import config


class VectorStoreManager:
    """ChromaDB 向量数据库管理器"""

    def __init__(self, collection_name: str = "enterprise_knowledge"):
        self.collection_name = collection_name
        self.embeddings = DoubaoEmbeddings()
        self._client = None
        self._vectorstore = None

    def _get_client(self):
        """获取 ChromaDB 客户端（优先 HTTP，回退到 Persistent）"""
        if self._client is not None:
            return self._client
        try:
            self._client = chromadb.HttpClient(
                host=config.CHROMA_HOST, port=config.CHROMA_PORT
            )
            self._client.heartbeat()
            logger.info(f"连接 ChromaDB 服务端: {config.CHROMA_HOST}:{config.CHROMA_PORT}")
        except Exception:
            logger.warning("ChromaDB 服务端不可用，使用本地嵌入式模式")
            self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        return self._client

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            client = self._get_client()
            self._vectorstore = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self._vectorstore

    def build_from_documents(self, documents: list[Document]) -> Chroma:
        """从文档列表构建向量数据库"""
        client = self._get_client()
        # 如果集合已存在则删除重建
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            client=client,
            collection_name=self.collection_name,
        )
        logger.info(f"向量数据库构建完成: {len(documents)} 个文档块")
        return self._vectorstore

    def add_documents(self, documents: list[Document]):
        """增量添加文档"""
        self.vectorstore.add_documents(documents)
        logger.info(f"增量添加: {len(documents)} 个文档块")

    def as_retriever(self, k: int = 4):
        """获取检索器"""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def delete_collection(self):
        """删除集合"""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._vectorstore = None
            logger.info(f"集合 '{self.collection_name}' 已删除")
        except Exception as e:
            logger.error(f"删除集合失败: {e}")

    def get_status(self) -> dict:
        """获取知识库状态"""
        try:
            count = self.vectorstore._collection.count()
            return {"collection": self.collection_name, "document_count": count}
        except Exception as e:
            return {"collection": self.collection_name, "document_count": 0, "error": str(e)}
