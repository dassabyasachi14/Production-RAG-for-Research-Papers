"""
Production RAG Application — Streamlit UI

Upload a research paper PDF, process it through the full RAG pipeline,
and ask questions grounded in the document's content with proper citations.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(STORAGE_DIR, "chroma_db"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", os.path.join(STORAGE_DIR, "bm25_indexes"))

# Ensure storage directories exist
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(BM25_INDEX_PATH).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Component initialisation (cached — loaded once per process)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading AI models (first run may take a few minutes)…")
def _load_components():
    """Initialise all heavy components once and cache them."""
    from src.generation.answer_generator import AnswerGenerator
    from src.generation.llm_client import LLMClient
    from src.generation.prompt_manager import PromptManager
    from src.indexing.bm25_store import BM25Store
    from src.indexing.embedder import Embedder
    from src.indexing.vector_store import VectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

    embedder = Embedder(api_key=GOOGLE_API_KEY)
    vector_store = VectorStore(CHROMA_DB_PATH)
    bm25_store = BM25Store(BM25_INDEX_PATH)
    bm25_store.load_all_indexes()

    retriever = HybridRetriever(vector_store, bm25_store, embedder)
    reranker = CrossEncoderReranker()

    prompt_manager = PromptManager()
    llm_client = LLMClient(api_key=GOOGLE_API_KEY)
    generator = AnswerGenerator(llm_client, prompt_manager)

    return {
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25_store": bm25_store,
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
        "llm_client": llm_client,
        "prompt_manager": prompt_manager,
    }


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, answer_obj}]
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = {}  # {doc_id: filename}
    if "active_doc_id" not in st.session_state:
        st.session_state.active_doc_id = None


# ---------------------------------------------------------------------------
# PDF processing pipeline
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_bytes: bytes,
    filename: str,
    components: dict,
    progress_bar,
    status_text,
) -> str:
    """
    Run the full ingestion pipeline on an uploaded PDF.

    Returns the new doc_id.
    """
    from src.document_processor.chunker import build_chunks
    from src.document_processor.image_processor import extract_and_describe_images
    from src.document_processor.pdf_extractor import extract_text_by_page
    from src.document_processor.table_extractor import extract_tables

    doc_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # 1. Extract text
        status_text.text("Extracting text…")
        progress_bar.progress(10)
        page_texts = extract_text_by_page(tmp_path)

        # 2. Extract and describe images
        status_text.text("Describing figures via Claude Vision…")
        progress_bar.progress(30)
        image_chunks = extract_and_describe_images(
            pdf_path=tmp_path,
            doc_id=doc_id,
            filename=filename,
            llm_client=components["llm_client"],
            prompt_manager=components["prompt_manager"],
        )

        # 3. Extract tables
        status_text.text("Extracting tables…")
        progress_bar.progress(50)
        table_chunks = extract_tables(tmp_path, doc_id, filename)

        # 4. Chunk all content
        status_text.text("Chunking content…")
        progress_bar.progress(60)
        all_chunks = build_chunks(doc_id, filename, page_texts, image_chunks, table_chunks)

        if not all_chunks:
            st.warning("No content could be extracted from this PDF.")
            return doc_id

        # 5. Embed
        status_text.text("Generating embeddings…")
        progress_bar.progress(75)
        texts = [c.content for c in all_chunks]
        embeddings = components["embedder"].embed(texts)

        # 6. Index in vector store and BM25
        status_text.text("Indexing…")
        progress_bar.progress(90)
        components["vector_store"].add_chunks(all_chunks, embeddings)
        components["bm25_store"].build_index(doc_id, all_chunks)

        progress_bar.progress(100)
        status_text.text(
            f"Done! Indexed {len(all_chunks)} chunks "
            f"({sum(1 for c in all_chunks if c.content_type == 'text')} text, "
            f"{sum(1 for c in all_chunks if c.content_type == 'image_description')} images, "
            f"{sum(1 for c in all_chunks if c.content_type == 'table')} tables)."
        )

    finally:
        os.unlink(tmp_path)

    return doc_id


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_citations(answer_obj):
    """Render an expandable citations panel below an answer."""
    if not answer_obj or not answer_obj.citations:
        return

    with st.expander(f"View {len(answer_obj.citations)} source(s)", expanded=False):
        for citation in answer_obj.citations:
            st.markdown(
                f"**[{citation.citation_index}]** `{citation.filename}` — "
                f"p.{citation.page_number}"
            )
            st.caption(citation.excerpt)
            st.divider()


def _render_retrieved_chunks_debug(answer_obj):
    """Show retrieved chunks in a debug expander (collapsed by default)."""
    if not answer_obj or not answer_obj.retrieved_chunks:
        return

    with st.expander("Debug: retrieved chunks", expanded=False):
        for chunk in answer_obj.retrieved_chunks:
            st.markdown(
                f"**Rank {chunk.retrieval_rank}** | "
                f"Score: `{chunk.retrieval_score:.4f}` | "
                f"`{chunk.content_type}` | "
                f"p.{chunk.page_number}"
                + (f" | {chunk.section}" if chunk.section else "")
            )
            st.text(chunk.content[:300] + ("…" if len(chunk.content) > 300 else ""))
            st.divider()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    _init_session_state()

    # --- API key guard ---
    if not GOOGLE_API_KEY:
        st.error(
            "**GOOGLE_API_KEY is not set.**\n\n"
            "Copy `.env.example` to `.env` and add your Google AI Studio API key "
            "(free at https://aistudio.google.com/apikey), then restart the app."
        )
        st.stop()

    components = _load_components()

    # -----------------------------------------------------------------------
    # Sidebar — document management
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("📄 Research RAG")
        st.caption(
            "Upload a research paper PDF and ask questions grounded in its content."
        )
        st.divider()

        # Prompt version indicator
        try:
            active_v = components["prompt_manager"].get_active_version()
        except Exception:
            active_v = "unknown"
        st.caption(f"Prompt version: `{active_v}`")
        st.divider()

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Research Paper",
            type=["pdf"],
            help="Upload a PDF to analyse. Multiple documents can be uploaded one at a time.",
        )

        if uploaded_file is not None:
            # Check if this file was already processed in this session
            already_processed = uploaded_file.name in st.session_state.processed_docs.values()

            if not already_processed:
                process_btn = st.button("Process Document", type="primary", use_container_width=True)
                if process_btn:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    with st.spinner("Processing PDF…"):
                        try:
                            doc_id = process_pdf(
                                pdf_bytes=uploaded_file.read(),
                                filename=uploaded_file.name,
                                components=components,
                                progress_bar=progress_bar,
                                status_text=status_text,
                            )
                            st.session_state.processed_docs[doc_id] = uploaded_file.name
                            st.session_state.active_doc_id = doc_id
                            st.session_state.messages = []  # Reset chat for new doc
                            st.success(f"✓ '{uploaded_file.name}' processed successfully.")
                        except Exception as exc:
                            st.error(f"Processing failed: {exc}")
                            logger.exception("PDF processing error")
            else:
                st.info(f"'{uploaded_file.name}' is already indexed.")

        # Indexed documents list
        if st.session_state.processed_docs:
            st.divider()
            st.subheader("Indexed Documents")
            for doc_id, fname in list(st.session_state.processed_docs.items()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    is_active = doc_id == st.session_state.active_doc_id
                    label = f"{'▶ ' if is_active else ''}{fname}"
                    if st.button(label, key=f"select_{doc_id}", use_container_width=True):
                        st.session_state.active_doc_id = doc_id
                        st.session_state.messages = []
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{doc_id}", help="Remove document"):
                        components["vector_store"].delete_document(doc_id)
                        components["bm25_store"].delete_document(doc_id)
                        del st.session_state.processed_docs[doc_id]
                        if st.session_state.active_doc_id == doc_id:
                            remaining = list(st.session_state.processed_docs.keys())
                            st.session_state.active_doc_id = remaining[0] if remaining else None
                            st.session_state.messages = []
                        st.rerun()

    # -----------------------------------------------------------------------
    # Main area — chat interface
    # -----------------------------------------------------------------------
    active_doc_id = st.session_state.active_doc_id
    active_fname = st.session_state.processed_docs.get(active_doc_id, "")

    if not active_doc_id:
        st.markdown(
            "## Welcome to Research RAG\n\n"
            "**Getting started:**\n"
            "1. Upload a research paper PDF using the sidebar\n"
            "2. Click **Process Document** to extract and index its content\n"
            "3. Ask questions — every answer will be grounded in citations\n\n"
            "_Supports text, tables, and figures (described via Claude Vision)_"
        )
        return

    st.header(f"📑 {active_fname}")
    st.caption(
        "Hybrid retrieval (BM25 + vector) · Cross-encoder reranking · Citation-enforced answers"
    )
    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("answer_obj"):
                _render_citations(msg["answer_obj"])
                _render_retrieved_chunks_debug(msg["answer_obj"])

    # Chat input
    if prompt := st.chat_input(f"Ask a question about '{active_fname}'…"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning…"):
                try:
                    # Hybrid retrieval
                    candidates = components["retriever"].retrieve(
                        query=prompt,
                        doc_id=active_doc_id,
                    )

                    # Cross-encoder reranking
                    reranked = components["reranker"].rerank(
                        query=prompt,
                        chunks=candidates,
                        top_k=8,
                    )

                    # Answer generation
                    answer_obj = components["generator"].generate(
                        query=prompt,
                        chunks=reranked,
                    )
                except Exception as exc:
                    st.error(f"An error occurred during generation: {exc}")
                    logger.exception("Answer generation error")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {exc}", "answer_obj": None}
                    )
                    st.stop()

            # Display answer
            if not answer_obj.is_grounded:
                st.warning(answer_obj.answer)
            else:
                st.markdown(answer_obj.answer)

            _render_citations(answer_obj)
            _render_retrieved_chunks_debug(answer_obj)

        # Persist to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_obj.answer,
                "answer_obj": answer_obj,
            }
        )

    _render_evaluation_section(components, active_doc_id, active_fname)


def _render_evaluation_section(components: dict, active_doc_id: str, active_fname: str):
    """Render the evaluation expander section below the chat interface."""
    import pandas as pd

    from src.evaluation.evaluator import Evaluator
    from src.evaluation.metrics import run_evaluation
    from src.evaluation.test_set_generator import TestSetGenerator

    st.divider()
    with st.expander("📊 Evaluate this document", expanded=False):
        st.markdown(
            "Automatically generate test questions from the document and score the "
            "RAG pipeline on **Faithfulness**, **Answer Relevancy**, and **Context Precision** "
            "using Gemini as an LLM judge."
        )

        n_q = st.slider(
            "Number of test questions", min_value=3, max_value=15, value=5, key="eval_n_q"
        )

        if st.button("Generate test questions", key="eval_gen_btn"):
            with st.spinner("Generating questions from document chunks…"):
                try:
                    gen = TestSetGenerator(
                        llm_client=components["llm_client"],
                        prompt_manager=components["prompt_manager"],
                        vector_store=components["vector_store"],
                    )
                    test_cases = gen.generate(doc_id=active_doc_id, n_questions=n_q)
                    st.session_state.eval_test_cases = test_cases
                    # Reset any previous report when questions are regenerated
                    st.session_state.pop("eval_report", None)
                    if not test_cases:
                        st.warning(
                            "No questions were generated. The document may not have enough "
                            "indexed text chunks, or the LLM response could not be parsed. "
                            "Try processing the document again or check the application logs."
                        )
                except Exception as exc:
                    st.error(f"Question generation failed: {exc}")
                    logger.exception("Test set generation error")

        # Editable question list
        if st.session_state.get("eval_test_cases"):
            st.markdown("**Review and edit questions before running:**")
            edited_questions = []
            for i, tc in enumerate(st.session_state.eval_test_cases):
                edited = st.text_input(
                    f"Q{i + 1}", value=tc.question, key=f"eval_q_{i}"
                )
                edited_questions.append(edited)

            if st.button("Run evaluation", key="eval_run_btn", type="primary"):
                # Apply any edits the user made
                for i, tc in enumerate(st.session_state.eval_test_cases):
                    st.session_state.eval_test_cases[i] = tc.model_copy(
                        update={"question": edited_questions[i]}
                    )

                progress = st.progress(0)
                status = st.empty()
                evaluator = Evaluator(
                    llm_client=components["llm_client"],
                    prompt_manager=components["prompt_manager"],
                )

                with st.spinner("Running pipeline and scoring answers…"):
                    try:
                        report = run_evaluation(
                            test_cases=st.session_state.eval_test_cases,
                            retriever=components["retriever"],
                            reranker=components["reranker"],
                            generator=components["generator"],
                            evaluator=evaluator,
                            doc_id=active_doc_id,
                            filename=active_fname,
                        )
                        st.session_state.eval_report = report
                        progress.progress(100)
                        status.empty()
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                        logger.exception("Evaluation run error")

        # Display results
        if st.session_state.get("eval_report"):
            report = st.session_state.eval_report
            st.markdown("### Results")

            # Aggregate metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Faithfulness",
                f"{report.mean_faithfulness:.2f}",
                help="Fraction of answer claims supported by context (1.0 = no hallucination)",
            )
            col2.metric(
                "Answer Relevancy",
                f"{report.mean_answer_relevancy:.2f}",
                help="How directly the answer addresses the question (1.0 = fully responsive)",
            )
            col3.metric(
                "Context Precision",
                f"{report.mean_context_precision:.2f}",
                help="Fraction of retrieved chunks that are relevant (1.0 = perfect retrieval)",
            )
            col4.metric(
                "Grounded",
                f"{report.n_grounded}/{report.n_questions}",
                help="Answers backed by the document vs. INSUFFICIENT_EVIDENCE declines",
            )

            st.markdown("### Per-question breakdown")
            rows = []
            for s in report.samples:
                rows.append(
                    {
                        "Question": s.question,
                        "Grounded": "✓" if s.is_grounded else "✗",
                        "Faithfulness": f"{s.faithfulness:.2f}" if s.faithfulness is not None else "—",
                        "Answer Relevancy": f"{s.answer_relevancy:.2f}" if s.answer_relevancy is not None else "—",
                        "Context Precision": f"{s.context_precision:.2f}" if s.context_precision is not None else "—",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
