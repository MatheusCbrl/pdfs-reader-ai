import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Leitor de PDF Licitações", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("PDF Licitações :books:")
  #  st.header("01. Coloque sua Chave API Key: ")
  #  api_key_input = st.text_input(
  #     "OpenAI API Key",
  #      type="password",
  #      placeholder="Cole sua Chave aqui (sk-...)",
  #      help="Pode pegar sua API Key aqui: https://platform.openai.com/account/api-keys.",  # noqa: E501
  #      value=os.environ.get("OPENAI_API_KEY", None)
  #       or st.session_state.get("OPENAI_API_KEY", ""),
  #      )
  #  st.session_state["OPENAI_API_KEY"] = api_key_input
    
  #  openai_api_key = st.session_state.get("OPENAI_API_KEY")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if not openai_api_key:
        st.warning(
        "Coloque a Sua API key. Pode conseguir ela em: "
        " https://platform.openai.com/account/api-keys.")

    st.header("02. Arraste ou Carregue os Arquivos: ")
    pdf_docs = st.file_uploader(
        label="Adicione seus PDFs aqui e clique em 'Processar'", accept_multiple_files=True)
    if st.button("Processar"):
        if not openai_api_key:
            st.warning("Coloque a Sua API key. Pode conseguir ela em: "
                       " https://platform.openai.com/account/api-keys.")
            st.stop()

        with st.spinner("Procesando... Pode demorar um pouco!"):
           # get pdf text
           raw_text = get_pdf_text(pdf_docs)

           # get the text chunks
           text_chunks = get_text_chunks(raw_text)

           # create vector store
           vectorstore = get_vectorstore(text_chunks)

           # create conversation chain
           st.session_state.conversation = get_conversation_chain(vectorstore)
    st.header("03. Pergunte sobre o documento que adicionou: ")
    user_question = st.text_input("Ex:. Quais as minhas licitaçõe? : ")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.markdown(
            "## Como Usar:\n"
            "1. Cole a sua chave [OpenAI API key](https://platform.openai.com/account/api-keys) abaixo🔑\n"  # noqa: E501
            "2. Carregue o pdf, docx, ou arquivo txt📄\n"
            "3. Pergunte o documento💬\n"
        )
        st.markdown("---")
        st.markdown("# About")
        with st.expander("Desenvolvedor"):
            st.markdown("📖Matheus Cabral\n\n"
                        "+55 54 999307783. ")        
           

if __name__ == '__main__':
    main()