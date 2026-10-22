
import streamlit as st
from pathlib import Path
from rag_engine import answer_question

# Configure the page FIRST
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📄",
    layout="centered",
)

# App title
st.title("📄 Document AI Assistant")
st.write(
    "Upload a PDF or Word document and ask questions about it."
)

st.divider()

# Upload document
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx"],
    help="Choose a PDF or DOCX file.",
)

# Ask question
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

# Run your existing RAG pipeline
if submitted:
    if uploaded_file is None:
        st.warning("Please upload a PDF or DOCX file first.")

    elif not user_query.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Reading your document and generating an answer..."
            ):
                answer, sources = answer_question(
                    file_bytes=uploaded_file.getvalue(),
                    file_name=Path(uploaded_file.name).name,
                    user_query=user_query.strip(),
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
