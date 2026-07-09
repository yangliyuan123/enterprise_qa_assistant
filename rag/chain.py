"""RAG 核心链 — 基于 LCEL 构建的检索增强生成系统"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import config


RAG_PROMPT_TEMPLATE = """你是一个专业的企业知识问答助手。请根据以下参考文档回答用户的问题。

要求：
1. 只基于提供的参考文档内容回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确告知用户
3. 回答要简洁、准确、有条理
4. 在回答末尾列出引用的文档来源

## 参考文档：
{context}

## 用户问题：
{question}

## 回答："""


class RAGAssistant:
    """基于 LCEL 的 RAG 问答助手"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            openai_api_key=config.DEEPSEEK_API_KEY,
            openai_api_base=config.DEEPSEEK_BASE_URL,
            temperature=0.3,
        )
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        def format_docs(docs):
            if not docs:
                return "暂无相关文档。"
            parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                parts.append(f"[文档{i} | 来源: {source}]\n{doc.page_content}")
            return "\n\n---\n\n".join(parts)

        chain = (
            RunnableParallel(
                context=(lambda x: format_docs(x["retrieved_docs"])),
                question=RunnablePassthrough(),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def ask(self, question: str) -> dict:
        """提问并获取回答"""
        docs = self.retriever.invoke(question)
        # 提取来源
        sources = []
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "未知来源")
            if src not in seen:
                sources.append(src)
                seen.add(src)

        answer = self.chain.invoke({
            "retrieved_docs": docs,
            "question": question,
        })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(docs),
        }

    def stream_ask(self, question: str):
        """流式提问"""
        docs = self.retriever.invoke(question)
        for chunk in self.chain.stream({
            "retrieved_docs": docs,
            "question": question,
        }):
            yield chunk
