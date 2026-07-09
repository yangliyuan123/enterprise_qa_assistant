"""多格式文档加载器 — 支持 PDF、Word、TXT、Markdown"""
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from utils.helpers import logger


class EnterpriseDocumentLoader:
    """企业文档加载器，支持多种格式的批量加载"""

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "text",
        ".pdf": "pdf",
        ".docx": "word",
    }

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str | Path) -> list[Document]:
        """加载单个文件，自动识别格式"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {list(self.SUPPORTED_EXTENSIONS)}")

        doc_type = self.SUPPORTED_EXTENSIONS[ext]

        if doc_type == "text":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif doc_type == "pdf":
            loader = PyPDFLoader(str(file_path))
        elif doc_type == "word":
            loader = UnstructuredWordDocumentLoader(str(file_path))

        docs = loader.load()
        # 注入来源元数据
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_path"] = str(file_path)
            doc.metadata["file_type"] = ext

        logger.info(f"加载文档: {file_path.name} → {len(docs)} 页/段")
        return docs

    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """批量加载目录下所有支持的文档"""
        dir_path = Path(dir_path)
        all_docs = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file in dir_path.glob(f"*{ext}"):
                try:
                    docs = self.load_file(file)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"加载 {file.name} 失败: {e}")

        logger.info(f"目录加载完成: {dir_path} → 共 {len(all_docs)} 个文档片段")
        return all_docs
