"""工具函数：日志、格式化输出"""
import logging
import sys
from datetime import datetime


def setup_logger(name: str = "enterprise_qa", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def format_answer(answer: str, sources: list[str] | None = None) -> str:
    """格式化问答结果"""
    lines = ["=" * 60, "📝 回答:", answer]
    if sources:
        lines.append("")
        lines.append("📚 参考来源:")
        for i, src in enumerate(sources, 1):
            lines.append(f"  [{i}] {src}")
    lines.append("=" * 60)
    return "\n".join(lines)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
