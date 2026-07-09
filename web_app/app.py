"""企业知识问答助手 — Streamlit Web 界面"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from document_loader.loader import EnterpriseDocumentLoader
from document_processor.splitter import DocumentSplitter
from vector_store.store import VectorStoreManager
from rag.chain import RAGAssistant
from memory.history import ConversationHistoryManager
import config


def build_knowledge_base():
    """从本地 data/documents/ 目录加载文档并构建知识库"""
    loader = EnterpriseDocumentLoader()
    splitter = DocumentSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    docs = loader.load_directory(config.DOCUMENTS_DIR)
    if not docs:
        return 0
    chunks = splitter.split(docs)
    st.session_state.vs_manager.build_from_documents(chunks)
    retriever = st.session_state.vs_manager.as_retriever(k=config.RETRIEVAL_K)
    st.session_state.assistant = RAGAssistant(retriever)
    st.session_state.documents_loaded = True
    return len(chunks)


# ========== 页面配置 ==========
st.set_page_config(
    page_title="企业知识问答助手",
    page_icon="🏢",
    layout="wide",
)

# ========== 初始化 Session State ==========
if "vs_manager" not in st.session_state:
    st.session_state.vs_manager = VectorStoreManager()
if "history" not in st.session_state:
    st.session_state.history = ConversationHistoryManager()
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "auto_init_done" not in st.session_state:
    st.session_state.auto_init_done = False

# ========== 自动加载文档 ==========
if not st.session_state.auto_init_done:
    status = st.session_state.vs_manager.get_status()
    if status.get("document_count", 0) == 0:
        with st.spinner("首次启动，正在自动加载知识库..."):
            count = build_knowledge_base()
            if count:
                st.success(f"知识库已就绪: {count} 个文档块")
            else:
                st.warning("未在 data/documents/ 中找到文档，请添加文档后点击「重新构建知识库」")
    else:
        retriever = st.session_state.vs_manager.as_retriever(k=config.RETRIEVAL_K)
        st.session_state.assistant = RAGAssistant(retriever)
        st.session_state.documents_loaded = True
    st.session_state.auto_init_done = True
    st.rerun()

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("🏢 企业知识问答助手")
    st.divider()

    # 文档管理
    st.subheader("📄 文档管理")
    doc_dir = config.DOCUMENTS_DIR
    if doc_dir.exists():
        files = list(doc_dir.iterdir())
        st.caption(f"知识库目录: {len(files)} 个文件")

    if st.button("🔄 重新构建知识库", use_container_width=True):
        with st.spinner("正在重新构建知识库..."):
            count = build_knowledge_base()
            if count:
                st.success(f"知识库重建完成: {count} 个文档块")
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("未找到任何文档")

    st.divider()

    # 知识库状态
    st.subheader("📊 知识库状态")
    status = st.session_state.vs_manager.get_status()
    st.metric("文档块数量", status.get("document_count", 0))

    st.divider()

    # 操作区
    st.subheader("⚙️ 操作")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清除对话", use_container_width=True):
            st.session_state.history.clear()
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("清空知识库", use_container_width=True):
            st.session_state.vs_manager.delete_collection()
            st.session_state.assistant = None
            st.session_state.documents_loaded = False
            st.session_state.messages = []
            st.session_state.auto_init_done = False
            st.rerun()

    st.divider()

    # 参数配置
    st.subheader("🔧 参数配置")
    retrieval_k = st.slider("检索文档数", 1, 10, config.RETRIEVAL_K)

# ========== 主区域 ==========
st.title("💬 企业知识问答")

# 聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 参考来源"):
                for src in msg["sources"]:
                    st.caption(f"- {src}")

# 输入区
if question := st.chat_input("请输入您的问题..."):
    if not st.session_state.documents_loaded or st.session_state.assistant is None:
        st.error("知识库未就绪，请点击左侧「重新构建知识库」")
    else:
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                assistant = st.session_state.assistant
                if retrieval_k != config.RETRIEVAL_K:
                    retriever = st.session_state.vs_manager.as_retriever(k=retrieval_k)
                    assistant = RAGAssistant(retriever)

                result = assistant.ask(question)
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📚 参考来源"):
                        for src in result["sources"]:
                            st.caption(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.session_state.history.add_turn(
            question, result["answer"], result["sources"]
        )
