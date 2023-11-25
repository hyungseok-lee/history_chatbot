import os
# from dotenv import load_dotenv
import streamlit as st
import tiktoken

# from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings



data_dir = "data/" 
openai_api_key = os.environ.get('OPENAI_API_KEY')

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def load_and_process_data(data_dir):
    doc_list = []
    pdf_list = sorted(os.listdir(data_dir))
    for idx, pdf_name in enumerate(pdf_list):
        if pdf_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, pdf_name))
            documents = loader.load_and_split()
            doc_list.extend(documents)
            print(f"{idx+1}/{len(pdf_list)} {pdf_name} loaded")
    print(f"Total {idx+1} documents loaded")
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo-1106', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain


st.set_page_config(page_title="HistoryChat", page_icon=":books:")
st.title("_History :red[QA Chat]_ :books:")

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
    
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                     "content": "역사에 관해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    doc_list = load_and_process_data(data_dir)
    text_chunks = get_text_chunks(doc_list)
    vetorestore = get_vectorstore(text_chunks)

    st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 
    st.session_state.processComplete = True





history = StreamlitChatMessageHistory(key="chat_messages")

# Chat logic
if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chain = st.session_state.conversation

        with st.spinner("Thinking..."):
            result = chain({"question": query})
            with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']
                
            response = result['answer']
            st.markdown(response)

            
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
