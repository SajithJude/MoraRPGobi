from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI


from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.llms import OpenAI

butt = st.button("load")

if butt:
    service_context = ServiceContext.from_defaults(
        llm=ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    )

    data = SimpleDirectoryReader(input_dir="data").load_data()
    index = VectorStoreIndex.from_documents(data, service_context=service_context)
    chat_engine = index.as_chat_engine(
        service_context=service_context, chat_mode="react", verbose=True
    )
    st.session_state.chat_engine = chat_engine
    st.success("success")

ques = st.text_input("Question")
ask = st.button("ask")

if ask:
    response = st.session_state.chat_engine.chat(ques)
    st.write(response)