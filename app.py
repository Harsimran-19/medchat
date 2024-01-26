from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import time
st.set_page_config(layout="wide")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
db = FAISS.load_local("faiss_index", embeddings)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 3})
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
temperature=0.75,
max_tokens=2000,
n_ctx = 4000,
top_p=1,
n_gpu_layers=10)

custom_prompt_template = """You are a medical practitioner who provides right medical information. Use the given following pieces of information to answer the user's question correctly. Add additional information from your knowledge base if necessary. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Current conversation: {chat_history}

Human: {question}
Only return the helpful answer below and nothing else.

AI Assistant:
"""
prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question','chat_history'])

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        ai_prefix="AI Assistant",
        return_messages=True
    ),
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# st.write(st.session_state.messages)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

prompt = st.chat_input("Say something")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking...",expanded=True):
            st_callback = StreamlitCallbackHandler(st.container())
            result = qa.invoke(input=prompt,callbacks=[st_callback])
            st.session_state.messages.append({"role":"assistant","content":result["answer"]})
            message_placeholder = st.empty()
            full_response = ""

        for chunk in result["answer"].split():
            full_response+=chunk+" "
            time.sleep(0.10)

            message_placeholder.markdown(full_response)
