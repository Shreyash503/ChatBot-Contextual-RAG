import streamlit as st
import tempfile
from rag_pipeline import load_documents, create_vector_store, build_qa_chain

st.set_page_config(page_title="📚 RAG Chatbot with ChromaDB + Groq", layout="wide")

st.title("🤖 Contextual RAG Chatbot (ChromaDB + Groq)")
st.markdown("Upload a document (PDF, TXT, DOCX) and ask contextual questions.")

uploaded_file = st.file_uploader("📂 Upload a file", type=["pdf", "txt", "docx"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"Loaded file: {uploaded_file.name}")

    # Process uploaded file
    docs = load_documents(file_path)
    vectorstore = create_vector_store(docs)
    qa_chain = build_qa_chain(vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User query
    query = st.text_input("💬 Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)

        # Save history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

    # Show chat history
    for role, text in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 {role}:** {text}")
        else:
            st.markdown(f"**🤖 {role}:** {text}")
