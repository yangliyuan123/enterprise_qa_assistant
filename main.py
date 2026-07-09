"""企业知识问答助手 — CLI 入口"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from document_loader.loader import EnterpriseDocumentLoader
from document_processor.splitter import DocumentSplitter
from vector_store.store import VectorStoreManager
from rag.chain import RAGAssistant
from rag.retriever import OptimizedRetriever
from memory.history import ConversationHistoryManager
from utils.helpers import format_answer, logger
import config


class App:
    """CLI 应用控制器"""

    def __init__(self):
        self.loader = EnterpriseDocumentLoader()
        self.splitter = DocumentSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vs_manager = VectorStoreManager()
        self.history = ConversationHistoryManager()
        self.optimized_retriever = OptimizedRetriever(self.vs_manager)
        self._assistant = None
        self._all_docs = []

    @property
    def assistant(self):
        if self._assistant is None:
            retriever = self.vs_manager.as_retriever(k=config.RETRIEVAL_K)
            self._assistant = RAGAssistant(retriever)
        return self._assistant

    def cmd_load(self, path: str):
        """加载文档到知识库"""
        target = Path(path)
        if target.is_dir():
            docs = self.loader.load_directory(target)
        else:
            docs = self.loader.load_file(target)

        if not docs:
            logger.warning("未找到任何文档")
            return

        chunks = self.splitter.split(docs)
        self._all_docs = chunks
        self.vs_manager.build_from_documents(chunks)
        self._assistant = None  # 重建 assistant 以使用新的检索器
        logger.info(f"知识库加载完成: {len(chunks)} 个文档块")

    def cmd_ask(self, question: str):
        """提问"""
        result = self.assistant.ask(question)
        self.history.add_turn(question, result["answer"], result["sources"])
        print(format_answer(result["answer"], result["sources"]))

    def cmd_status(self):
        """查看知识库状态"""
        status = self.vs_manager.get_status()
        print(f"\n知识库状态:")
        print(f"  集合名称: {status['collection']}")
        print(f"  文档块数: {status['document_count']}")
        print(f"  对话轮数: {len(self.history) // 2}")

    def cmd_history(self):
        """查看对话历史"""
        if len(self.history) == 0:
            print("暂无对话历史")
            return
        for i, turn in enumerate(self.history.get_all()):
            if "question" in turn:
                print(f"\n[Q{i // 2 + 1}] {turn['question']}")
            if "answer" in turn:
                print(f"[A{i // 2 + 1}] {turn['answer'][:100]}...")

    def cmd_clear(self):
        """清除对话历史"""
        self.history.clear()
        print("对话历史已清除")

    def cmd_export(self, path: str = "conversation_history.json"):
        """导出对话记录"""
        self.history.export(path)
        print(f"已导出至: {path}")


def print_banner():
    print("""
╔══════════════════════════════════════════════╗
║        🏢 企业知识问答助手                    ║
║        Enterprise Knowledge Q&A Assistant    ║
╚══════════════════════════════════════════════╝
命令: load <路径> | ask <问题> | status | history | clear | export | quit
""")


def main():
    app = App()
    print_banner()

    while True:
        try:
            raw = input("\n> ").strip()
            if not raw:
                continue
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if raw.lower() == "quit":
            print("再见！")
            break

        if raw.lower() == "status":
            app.cmd_status()
        elif raw.lower() == "history":
            app.cmd_history()
        elif raw.lower() == "clear":
            app.cmd_clear()
        elif raw.lower().startswith("export"):
            path = raw[6:].strip() or "conversation_history.json"
            app.cmd_export(path)
        elif raw.lower().startswith("load"):
            path = raw[4:].strip()
            if not path:
                print("用法: load <文件路径或目录>")
            else:
                app.cmd_load(path)
        elif raw.lower().startswith("ask"):
            question = raw[3:].strip()
            if not question:
                print("用法: ask <问题>")
            else:
                app.cmd_ask(question)
        else:
            # 默认当作提问处理
            app.cmd_ask(raw)


if __name__ == "__main__":
    main()
