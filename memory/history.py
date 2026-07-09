"""对话历史管理 — 短期记忆 + 长期向量存储记忆"""
import json
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from utils.helpers import logger


class ConversationHistoryManager:
    """对话历史管理器"""

    def __init__(self, max_recent: int = 10):
        self.max_recent = max_recent
        self._history: list[dict] = []

    def add_turn(self, question: str, answer: str, sources: list[str] | None = None):
        """添加一轮对话"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or [],
        }
        self._history.append(turn)
        # 保留最近 N 轮
        if len(self._history) > self.max_recent * 2:
            self._history = self._history[-self.max_recent * 2:]

    def get_recent_context(self, n: int | None = None) -> str:
        """获取最近 N 轮对话作为上下文"""
        n = n or self.max_recent
        recent = self._history[-(n * 2):]
        if not recent:
            return ""
        parts = []
        for i, turn in enumerate(recent):
            role = "用户" if i % 2 == 0 else "助手"
            content = turn.get("question") or turn.get("answer")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def get_all(self) -> list[dict]:
        return self._history

    def clear(self):
        self._history.clear()
        logger.info("对话历史已清除")

    def export(self, file_path: str | Path):
        """导出对话记录为 JSON"""
        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)
        logger.info(f"对话记录已导出: {file_path}")

    def as_chat_messages(self) -> list[tuple[str, str]]:
        """转为 (role, content) 格式，用于 LLM 上下文"""
        messages = []
        for turn in self._history:
            if "question" in turn:
                messages.append(("user", turn["question"]))
            if "answer" in turn:
                messages.append(("assistant", turn["answer"]))
        return messages

    def __len__(self):
        return len(self._history)
