from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config import APP_TITLE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, DEFAULT_TOP_K, DEMO_TRAINING_DATA
from models.finetuning import FineTuningSetup
from models.llm import LLMManager
from rag.chunking import build_chunks_from_documents
from rag.document_loader import extract_text_from_files
from rag.embeddings import EmbeddingModel
from rag.vector_store import FAISSVectorStore


st.set_page_config(page_title=APP_TITLE, page_icon="🤖")


@st.cache_resource
def get_llm_manager(model_name: str = DEFAULT_MODEL):
    manager = LLMManager(model_name=model_name)
    manager.load()
    return manager


@st.cache_resource
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return EmbeddingModel(model_name)


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {chunk['filename']} page {chunk.get('page_number', 'unknown')}]:\n{chunk['text']}" for chunk in chunks)
    return (
        "You are a careful assistant. Use only the supplied context to answer the user question. "
        "If the answer is not in the context, say that the information is not present in the uploaded documents. "
        "Do not invent sources or citations.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def initialize_session_state():
    defaults = {
        "uploaded_documents": [],
        "vector_store": None,
        "embedding_model": None,
        "rag_chunks": [],
        "fine_tuned_available": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

st.title(APP_TITLE)
st.write("A beginner-friendly GenAI lab project using an open-source LLM, RAG, and LoRA/QLoRA.")

base_tab, rag_tab, finetuning_tab, compare_tab, about_tab = st.tabs([
    "Chat with Base LLM",
    "Chat with Your Documents",
    "Fine-Tuning",
    "Compare Models",
    "About the Project",
])

with base_tab:
    st.subheader("Base model chat")
    question = st.text_area("Ask a question without documents:", height=120)
    if st.button("Generate answer") and question.strip():
        with st.spinner("Loading model..."):
            llm = get_llm_manager()
        answer = llm.generate(question, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
        st.markdown("**Question:**")
        st.write(question)
        st.markdown("**Answer:**")
        st.write(answer)
        st.info("Uploaded documents were not used as retrieval context for this response.")

with rag_tab:
    st.subheader("Upload documents and ask grounded questions")
    uploaded_files = st.file_uploader("Upload PDFs, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if uploaded_files:
        saved_files = []
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = upload_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getvalue())
            saved_files.append(str(file_path))
        st.session_state["uploaded_documents"] = saved_files
        st.success(f"{len(saved_files)} file(s) uploaded successfully.")

    if st.button("Build document index") and st.session_state["uploaded_documents"]:
        docs = extract_text_from_files(st.session_state["uploaded_documents"])
        valid_docs = [doc for doc in docs if not doc.get("error")]
        if not valid_docs:
            st.warning("No valid documents were found. Please upload readable files.")
        else:
            chunks = build_chunks_from_documents(valid_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            st.session_state["rag_chunks"] = chunks
            embedding_model = get_embedding_model()
            embeddings = embedding_model.encode([chunk["text"] for chunk in chunks])
            vector_store = FAISSVectorStore(embeddings.shape[1])
            vector_store.add(embeddings, chunks)
            st.session_state["vector_store"] = vector_store
            st.session_state["embedding_model"] = embedding_model
            st.success(f"Indexed {len(chunks)} chunks from {len(valid_docs)} document(s).")

    question = st.text_area("Ask a question about your documents:", height=120, key="rag_question")
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=8, value=DEFAULT_TOP_K)
    if st.button("Generate RAG answer") and question.strip() and st.session_state["vector_store"] is not None:
        with st.spinner("Retrieving and generating..."):
            embedding_model = st.session_state["embedding_model"]
            query_embedding = embedding_model.encode([question])[0]
            matches = st.session_state["vector_store"].search(query_embedding, top_k)
            retrieved_chunks = [match["metadata"] for match in matches]
            if not retrieved_chunks:
                st.warning("No relevant passages were retrieved from the uploaded documents.")
            else:
                prompt = build_rag_prompt(question, retrieved_chunks)
                llm = get_llm_manager()
                response = llm.generate(prompt, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
                st.markdown("**Question:**")
                st.write(question)
                st.markdown("**Answer:**")
                st.write(response)
                st.markdown("**Retrieved chunks:**")
                for chunk in retrieved_chunks:
                    st.markdown(f"- {chunk['filename']} (page {chunk.get('page_number', 'unknown')})")
                    st.write(chunk['text'])
                st.caption(f"Retrieved {len(retrieved_chunks)} chunks.")
    elif st.button("Generate RAG answer") and not st.session_state["vector_store"]:
        st.warning("Build the document index before asking a question.")

with finetuning_tab:
    st.subheader("LoRA / QLoRA fine-tuning")
    st.markdown("- LoRA adds trainable low-rank adapters to frozen model weights.")
    st.markdown("- QLoRA quantizes the base model for lower memory usage while still training adapters.")
    st.markdown("- Full fine-tuning updates all model parameters; LoRA and QLoRA are more memory-efficient.")

    dataset_file = st.file_uploader("Upload training data (JSONL)", type=["jsonl"], help="Each line must contain an instruction and response pair.")
    if dataset_file is not None:
        dataset_path = Path("data/temp_training.jsonl")
        dataset_path.parent.mkdir(exist_ok=True)
        dataset_path.write_bytes(dataset_file.getvalue())
        st.session_state["training_dataset_path"] = str(dataset_path)
        st.success("Training data uploaded.")

    if st.button("Validate dataset") and "training_dataset_path" in st.session_state:
        setup = FineTuningSetup(DEFAULT_MODEL)
        rows = setup.validate_dataset(st.session_state["training_dataset_path"])
        st.success(f"Dataset valid. {len(rows)} example(s) loaded.")

    rank = st.slider("LoRA rank", min_value=4, max_value=32, value=8)
    alpha = st.slider("LoRA alpha", min_value=8, max_value=32, value=16)
    dropout = st.slider("LoRA dropout", min_value=0.0, max_value=0.3, step=0.05, value=0.05)
    if st.button("Run demonstration training setup"):
        setup = FineTuningSetup(DEFAULT_MODEL)
        setup.build_lora_config(rank=rank, alpha=alpha, dropout=dropout)
        st.info("A real train run requires sufficient GPU memory and may take time. This app demonstrates the setup and stores adapter configuration for later use.")
        demo_path = DEMO_TRAINING_DATA
        if demo_path.exists():
            rows = setup.validate_dataset(demo_path)
            st.success(f"Demo dataset validated with {len(rows)} example(s).")
        else:
            st.warning("Demo dataset not found.")

    st.markdown("### Example dataset")
    with open(DEMO_TRAINING_DATA, "r", encoding="utf-8") as handle:
        st.code(handle.read())

with compare_tab:
    st.subheader("Compare base, RAG, and fine-tuned responses")
    evaluation_questions = [
        "What is a large language model?",
        "What is retrieval-augmented generation?",
        "Why is LoRA useful?",
    ]
    if st.button("Generate comparison"):
        llm = get_llm_manager()
        base_answers = [llm.generate(q, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) for q in evaluation_questions]
        rag_answers = []
        if st.session_state["vector_store"] is not None:
            for q in evaluation_questions:
                embedding_model = st.session_state["embedding_model"]
                query_embedding = embedding_model.encode([q])[0]
                matches = st.session_state["vector_store"].search(query_embedding, DEFAULT_TOP_K)
                retrieved_chunks = [match["metadata"] for match in matches]
                prompt = build_rag_prompt(q, retrieved_chunks)
                rag_answers.append(llm.generate(prompt, max_new_tokens=DEFAULT_MAX_NEW_TOKENS))
        else:
            rag_answers = ["RAG not available until documents are indexed."] * len(evaluation_questions)

        fine_tuned_answers = ["Fine-tuned adapter not available yet."] * len(evaluation_questions)
        if st.session_state.get("fine_tuned_available"):
            fine_tuned_answers = [llm.generate(q, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) for q in evaluation_questions]

        comparison_data = []
        for idx, question in enumerate(evaluation_questions):
            comparison_data.append({
                "Question": question,
                "Base LLM": base_answers[idx],
                "RAG": rag_answers[idx],
                "Fine-tuned": fine_tuned_answers[idx],
            })
        st.dataframe(comparison_data, use_container_width=True)
        st.info("RAG and fine-tuning solve different problems: retrieval grounds answers in user documents, while fine-tuning changes the model's behavior on a task.")

with about_tab:
    st.header("About the project")
    st.markdown("""
    This project is a college-friendly GenAI demo that shows three common patterns:

    1. Base LLM chat: ask the model a direct question without document retrieval.
    2. RAG: fetch relevant document chunks, pass them to the model, and answer using the retrieved context.
    3. Fine-tuning: adapt the model with LoRA or QLoRA using a small JSONL dataset.

    RAG is best when you need answers grounded in personal or private documents. Fine-tuning is best when you want a model to follow a specific pattern or task more closely across many inputs. Both can be useful together, but they are not interchangeable.
    """)
    st.markdown("### Why chunking, embeddings, and FAISS are needed")
    st.markdown("- Chunking breaks long documents into manageable pieces.")
    st.markdown("- Embeddings convert chunks into numeric vectors that capture semantics.")
    st.markdown("- FAISS quickly finds document chunks similar to the user's query.")
    st.markdown("- This allows the model to answer with grounded context rather than guessing from general world knowledge.")
