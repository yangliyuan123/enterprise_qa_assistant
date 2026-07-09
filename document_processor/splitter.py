"""文档分割策略 — 智能切分长文档"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from utils.helpers import logger


class DocumentSplitter:
    """文档分割器，支持通用分割和 Markdown 结构分割"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """通用文档分割"""
        chunks = self._text_splitter.split_documents(documents)
        logger.info(f"文档分割: {len(documents)} → {len(chunks)} 个块 (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def split_markdown(self, text: str, metadata: dict | None = None) -> list[Document]:
        """按 Markdown 标题结构分割"""
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        docs = md_splitter.split_text(text)
        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)
        return docs

    def split_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """直接分割文本"""
        chunks = self._text_splitter.create_documents([text])
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        return chunks
