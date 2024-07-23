import streamlit as st
#from dotenv import load_dotenv
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

def get_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore ):
   
    llm = ChatOpenAI()

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
    #load_dotenv()
    st.set_page_config(page_title="Leitor de PDF Licitações",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("PDF Licitações :books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    with st.sidebar:
        st.markdown(
            "## Como Usar:\n"
            "1. Carregue o pdf📄\n"
            "2. Pergunte o que quiser sobre o documento💬\n"
        )
        st.markdown("---")
        st.markdown("# Sobre")
        with st.expander("Eng. IA 📖"):
            st.markdown("Matheus Cabral\n\n"
                    "+55 54 999307783. ")
    # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
    # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
    openai_api_key = st.secrets['openai']["OPENAI_API_KEY"]
    #openai_api_key = st.text_input(
    #    "OpenAI API Key", 
    #    type="password",
    #    placeholder="Cole sua Chave aqui (sk-...)",
    #    help="Pode pegar sua API Key aqui: https://platform.openai.com/account/api-keys."
    #)
    if not openai_api_key:
        st.info("A Usa API key da OpenAI para contunuar.", icon="🗝️")
    else:
        st.header("02. Arraste ou Carregue os Arquivos: ")
        pdf_docs = st.file_uploader(label="Adicione seus PDFs aqui e clique em 'Processar'", accept_multiple_files=True)
        if not pdf_docs:
            st.info("Não inseriu um arquivo, insira-o para continuar! ")
        else:
            with st.spinner("Procesando... Pode demorar um pouco!"):
            # get pdf text
             raw_text = get_pdf_text(pdf_docs)
            # get the text chunks
             text_chunks = get_text_chunks(raw_text)
            # create vector store
             vectorstore = get_vectorstore(text_chunks, openai_api_key)
            # create conversation chain
             st.session_state.conversation = get_conversation_chain(vectorstore)
             st.header("03. Pergunte sobre o documento que adicionou: ")
             user_question = st.text_input(" ",
                                           help="Ex:. Quais as minhas licitações? : ",
                                           placeholder="Digite e pressione 'Enter' ")
                 #if pdf_docs and user_question and not api_key_input:
                 #st.info("Ops! Insira sua API ao lado para continuar")
             if pdf_docs and user_question:
                 handle_userinput(user_question)

if __name__ == '__main__':
    main()