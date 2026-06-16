
import streamlit as st
import tempfile
import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.embeddings.embedder import (
    ingest_financial_folder, ingest_regulatory_folder,
    load_financial_vectorstore, load_regulatory_vectorstore,
    add_documents_to_financial, add_documents_to_regulatory,
    get_indexed_sources
)
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import create_chunks
from src.ingestion.metadata_parser import parse_financial_metadata, parse_regulatory_metadata
from src.agent.tools import initialize_tools
from src.agent.graph import build_agent
from src.config import config


st.set_page_config(
    page_title="Findoc Agent",
    page_icon="📊",
    layout="wide")

st.title("Findoc Agent")
st.caption("Análisis conversacional de documentos financieros y normativa regulatoria")

# Estado de sesión

if "financial_vs" not in st.session_state:
    st.session_state.financial_vs = None
if "regulatory_vs" not in st.session_state:
    st.session_state.regulatory_vs = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    

# Inicialización automática si ya existen las colecciones en disco
if st.session_state.financial_vs is None:
    financial_dir = Path(config.CHROMA_FINANCIAL_DIR)
    regulatory_dir = Path(config.CHROMA_REGULATORY_DIR)

    if financial_dir.exists() and regulatory_dir.exists():
        with st.spinner("Cargando colecciones existentes..."):
            st.session_state.financial_vs = load_financial_vectorstore()
            st.session_state.regulatory_vs = load_regulatory_vectorstore()
            initialize_tools(st.session_state.financial_vs, st.session_state.regulatory_vs)
            st.session_state.agent = build_agent()

# Sidebar
with st.sidebar:
    st.header("Documentos")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Indexar carpetas", type="primary", use_container_width=True):
            with st.spinner("Indexando data/financial/ y data/regulatory/..."):
                st.session_state.financial_vs = ingest_financial_folder()
                st.session_state.regulatory_vs = ingest_regulatory_folder()
                initialize_tools(st.session_state.financial_vs, st.session_state.regulatory_vs)
                st.session_state.agent = build_agent()
                st.success("Colecciones listas")

    st.divider()
    st.caption("Añadir documentos individuales:")

    collection_type = st.selectbox("Tipo de colección", ["Financiero", "Regulatorio"])

    uploaded_file = st.file_uploader("Sube un PDF", type="pdf")

    if uploaded_file and st.button("Indexar PDF"):
        with st.spinner(f"Procesando {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                pages = load_pdf(tmp_path)

                if collection_type == "Financiero":
                    metadata = parse_financial_metadata(uploaded_file.name)
                    for page in pages:
                        page.update(metadata)
                    chunks = create_chunks(pages)

                    if st.session_state.financial_vs is None:
                        from src.embeddings.embedder import create_financial_vectorstore
                        st.session_state.financial_vs = create_financial_vectorstore(chunks)
                    else:
                        add_documents_to_financial(chunks)

                else:
                    metadata = parse_regulatory_metadata(uploaded_file.name)
                    for page in pages:
                        page.update(metadata)
                    chunks = create_chunks(pages)

                    if st.session_state.regulatory_vs is None:
                        from src.embeddings.embedder import create_regulatory_vectorstore
                        st.session_state.regulatory_vs = create_regulatory_vectorstore(chunks)
                    else:
                        add_documents_to_regulatory(chunks)

                initialize_tools(st.session_state.financial_vs, st.session_state.regulatory_vs)
                st.session_state.agent = build_agent()
                st.success(f"✓ {uploaded_file.name} indexado")

            finally:
                os.unlink(tmp_path)

    # Mostrar fuentes indexadas
    if st.session_state.financial_vs:
        st.divider()
        st.caption("Financieros indexados:")
        for src in get_indexed_sources(st.session_state.financial_vs):
            st.text(f"• {src}")

    if st.session_state.regulatory_vs:
        st.caption("Regulatorios indexados:")
        for src in get_indexed_sources(st.session_state.regulatory_vs):
            st.text(f"• {src}")

# Área de chat
if not st.session_state.agent:
    st.info("Usa el botón 'Indexar carpetas' en el panel lateral para inicializar el sistema.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregunta sobre informes financieros o normativa regulatoria..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando..."):
                result = st.session_state.agent.invoke({"input": prompt})
                answer = result["output"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

