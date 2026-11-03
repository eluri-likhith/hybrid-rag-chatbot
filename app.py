
import streamlit as st
import hashlib
from pathlib import Path

from rag_engine import (
    answer_question,
    build_retrieval_components
)

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Document AI Assistant")
st.write(
    "Upload a PDF or Word document and ask questions about it."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx"],
    help="Choose a PDF or DOCX file.",
)

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()
    file_name = Path(uploaded_file.name).name

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        st.session_state.pop("vector_store", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("bm25_index", None)
        st.session_state["file_hash"] = file_hash

with st.form("question_form"):

    user_query = st.text_input(
        "Your question",
        placeholder="Example: What is the annual leave policy?",
    )

    submitted = st.form_submit_button(
        "Get Answer",
        type="primary",
        use_container_width=True,
    )

if submitted:

    if uploaded_file is None:
        st.warning("Please upload a PDF or DOCX file.")

    elif not user_query.strip():
        st.warning("Please enter your question.")

    else:
        try:

            file_bytes = uploaded_file.getvalue()
            file_name = Path(uploaded_file.name).name

            file_hash = hashlib.sha256(file_bytes).hexdigest()

            with st.spinner("Processing your document..."):

                if "vector_store" not in st.session_state:

                    vector_store, chunks, bm25_index = (
                        build_retrieval_components(
                            file_bytes,
                            file_name
                        )
                    )

                    st.session_state["vector_store"] = vector_store
                    st.session_state["chunks"] = chunks
                    st.session_state["bm25_index"] = bm25_index
                    st.session_state["file_hash"] = file_hash

            with st.spinner("Searching and generating your answer..."):

                answer, sources = answer_question(
                    file_name=file_name,
                    user_query=user_query.strip(),
                    vector_store=st.session_state["vector_store"],
                    chunks=st.session_state["chunks"],
                    bm25_index=st.session_state["bm25_index"],
                )

            st.subheader("Answer")

            with st.container(border=True):
                st.markdown(answer)

            st.subheader("Sources")

            with st.container(border=True):

                if sources:
                    for source in sources:
                        st.write(f"📄 {source}")
                else:
                    st.write("No source details available.")

        except Exception as error:
            st.error("Unable to process your request.")
            st.exception(error)
