


import streamlit as st

from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.llms import OpenAI


from llama_index.prompts  import Prompt

butt = st.button("load")

if butt:


    custom_prompt = Prompt("""\
    I am a novice in the content of the text in the previous prompt. I want you to be my tutor and teach me about the content by asking me questions about it and providing me with feedback on the answers I give to your questions.
    We will do this in a loop with the following steps:

    (1) You ask me a question about the text
    (2) I respond to your question
    (3) You give me feedback on my response.

    Question Rule 1: Do not ask questions about very small details such as specific numbers.
    Question Rule 2: When you run out of questions, let me know this and then end the quiz

    <Chat History> 
    {chat_history}

    <Follow Up Message>
    {question}

    <Standalone question>
    """)

    # list of (human_message, ai_message) tuples
    custom_chat_history = [
        (
           'Hello assistant, we are having a insightful quiz this document.', 
         'Okay, sounds good.'
        )
    ]
    service_context = ServiceContext.from_defaults(
        llm=ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    )

    data = SimpleDirectoryReader(input_dir="data").load_data()
    index = VectorStoreIndex.from_documents(data, service_context=service_context)
    # query_engine = index.as_query_engine()
    # chat_engine = CondenseQuestionChatEngine.from_defaults(
    #     query_engine=query_engine, 
    #     condense_question_prompt=custom_prompt,
    #     chat_history=custom_chat_history,
    #     verbose=True
    # )
    chat_engine = index.as_chat_engine(
        condense_question_prompt=custom_prompt,
    chat_mode='condense_question', 
    verbose=True
        )
    st.session_state.chat_engine = chat_engine
    st.success("success")

ques = st.text_input("Question")
ask = st.button("ask")

if ask:
    response = st.session_state.chat_engine.chat(ques)
    st.write(response.response)