import streamlit as st

from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.llms import OpenAI


from llama_index import Prompt
context_str = """(1) You ask me a question about the text

(2) I respond to your question

(3) You give me feedback on my response.
"""
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
custom_prompt = Prompt(template)

butt = st.button("load")

if butt:
    service_context = ServiceContext.from_defaults(
        llm=ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    )

    data = SimpleDirectoryReader(input_dir="data").load_data()
    index = VectorStoreIndex.from_documents(data, service_context=service_context)
    chat_engine = index.as_chat_engine(
        service_context=service_context, chat_mode="react",text_qa_template=custom_prompt, verbose=True
    )
    st.session_state.chat_engine = chat_engine
    st.success("success")

ques = st.text_input("Question")
ask = st.button("ask")

if ask:
    response = st.session_state.chat_engine.chat(f"Use the tool to answer:{ques}")
    st.write(response)