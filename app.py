from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("https://github.com/harshitv804/MedChat/assets/100853494/0aa18d7e-5305-4d8e-89d8-09fffce1589e")

st.markdown(
    """
    <style>
div.stButton > button:first-child {
    background-color: #ffd0d0;
}
div.stButton > button:active {
    background-color: #ff6262;
}

   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
db = FAISS.load_local("faiss_index", embeddings)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 3})

llm = LlamaCpp(
model_path="stablelm-zephyr-3b.Q4_K_M.gguf",
temperature=0.75,
max_tokens=2000,
n_ctx = 4000,
top_p=1)


custom_prompt_template = """You are a medical practitioner who provides right medical information. Use the given following pieces of information to answer the user's question correctly. If you don't know the answer, just say that you don't know, don't try to make up an answer. Utilize the provided knowledge base and search for relevant information. Follow and answer the question format closely. Give only the important information. The information should be abstract, high quality content and comprehensive.

Context: {context}

History: {chat_history}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate. Consult a qualified doctor for accurate advice._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})
