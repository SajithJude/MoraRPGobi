# import streamlit as st
# from llama_index.llms import OpenAI, ChatMessage
# from typing import List
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# import difflib

# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# stemmer = PorterStemmer()

# class TutorAgent:
#     def __init__(self, chat_history: List[ChatMessage] = []):
#         self._llm = llm
#         self._chat_history = chat_history
#         self.expected_answer = ''
#         self.text = ''

#     def reset(self):
#         self._chat_history = []

#     def generate_question(self) -> str:
#         self.reset()
#         message = self._llm.chat([ChatMessage(role="system", content=f"Generate a broad question about the following text: {self.text} and provide the expected answer")])
#         split_message = message.message.content.split('\n')
#         self.expected_answer = split_message[1]
#         st.write(split_message[1])
        
#         return split_message[0]

#     # Initialize the stemmer

#     def get_score(self, user_answer: str) -> float:
#         # use self.expected_answer directly in this method
#         sequence_matcher = difflib.SequenceMatcher(None, self.expected_answer, user_answer)
#         score = sequence_matcher.ratio()
#         return score



#     def compare_answers(self, user_answer: str) -> float:
#         return self.get_score(self.expected_answer, user_answer)

#     def give_feedback(self, answer: str) -> str:
#         self._chat_history.append(ChatMessage(role="user", content=answer))

#         feedback_instructions = """
#         Please provide detailed feedback based on the following principles:
        
#         Do not give away the correct answer if the answer is incorrect or only partly correct.
#         Make sure to mention the principle numbers that are relevant to your feedback.
#         If the answer is incorrect or only partly correct, guide the user to try again.

#         Expected answer: {self.expected_answer}
#         User's answer: {answer}
#         """

#         self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

#         message = self._llm.chat(self._chat_history)
#         feedback = message.message.content

#         score = self.get_score(answer)
#         if score < 0.0:  # replace with your threshold
#             self._chat_history = []  # clear chat history for new question
#             next_question = self.generate_question(text)
#             feedback += f"\n\nYour score was below the threshold. Here's your next question: {next_question}"

#         return feedback

# tutor = TutorAgent()

# st.title("AI Tutor")
# tutor.text = st.text_area("Input text for learning:", "Enter text here...")

# if st.button("Start learning session"):
#     question = tutor.generate_question()
#     st.write("Question: ", question)
#     st.write("Provide your answer and press 'Submit Answer' when ready.")

# answer = st.text_input("Your answer:")
# if st.button("Submit Answer"):
#     feedback = tutor.give_feedback(answer)
#     st.write("Feedback: ", feedback)

# # Display chat history on sidebar
# st.sidebar.header("Chat History")
# for message in tutor._chat_history:
#     if message.role == 'system':
#         st.sidebar.markdown(f"**System**: question")
#     else:
#         st.sidebar.markdown(f"**You**: {message.content}")


import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List
from nltk.translate.bleu_score import sentence_bleu


from llama_index import (
    GPTVectorStoreIndex, Document, SimpleDirectoryReader,
    QuestionAnswerPrompt, LLMPredictor, ServiceContext
)


# from langchain import OpenAI
from tempfile import NamedTemporaryFile
from llama_index import download_loader

import openai
import os
from pathlib import Path
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
PDFReader = download_loader("PDFReader")

from llama_index import Prompt

# loader = PDFReader()

openai.api_key = os.getenv("OPENAI_API_KEY")




llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")






def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    if "index" not in st.session_state:
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        query_engine = index.as_query_engine()
        st.session_state.index = query_engine
    # st.session_state.index = index
    return st.session_state.index






class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history
        self.score_threshold = 0.7  # adjust this as per your requirements
        self.expected_answer = ""  # Initialize expected_answer attribute

    def reset(self):
        self._chat_history = []

    def extract_keywords(self, text: str) -> List[str]:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Please list 10 keywords or topics from the following text: {text}")])
        keywords = message.message.content.split('\n')  # Assuming the model returns a newline-separated list
        return keywords

    def generate_question_answer(self, keyword: str) -> (str, str):
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a question about the topic: {keyword} with the answer separated by a newline.")])
        question, expected_answer = message.message.content.split('\n')  # Assuming the model returns question and answer separated by a newline
        # st.write(question)
        # st.write(expected_answer)

        self.expected_answer = expected_answer
        return question, expected_answer

    def give_feedback(self, user_answer: str) -> (str, float):
        self._chat_history.append(ChatMessage(role="user", content=user_answer))

        feedback_instructions = """
        Please provide detailed feedback based on the following principles:
       
        Do not give away the correct answer if the answer is incorrect or only partly correct.
        Make sure to mention the principle numbers that are relevant to your feedback.
        If the answer is incorrect or only partly correct, mention the areas to improve.

        Expected answer: {self.expected_answer}
        User's answer: {answer}
        """
        
        self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

        message = self._llm.chat(self._chat_history)
        feedback = message.message.content

        score = self.get_score(user_answer, self.expected_answer)
        return feedback, score

    def get_score(self, user_answer: str, expected_answer: str) -> float:
        
        Prompt = """
        evaluate on a general understanding of how well the user's answer aligns with the expected answer.
        In a simple format, I could use a scale of 1-10 where 1 signifies no alignment with the expected answer and 10 signifies perfect alignment.
        Return only a number between 1 and 10 in the response.
        Expected answer: {expected_answer}
        User's answer: {user_answer}
        """
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a question about the topic: {keyword} with the answer separated by a newline.")])
        score = message.message.content
        return score


# try:

tutor = TutorAgent()

st.title("AI Tutor")







# text = st.text_area("Input text for learning:", "Enter text here...")

# if "keywords" not in st.session_state:
#     keywords = []


# if st.button("Index topics"):

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # if "index" not in st.session_state:
    st.session_state.index = process_pdf(uploaded_file)



        # st.success("Index created successfully")
    # keywords = tutor.extract_keywords(text)
# if "keywords" not in st.session_state:
    
    res  = st.session_state.index.query("Please list 10 keywords or topics from the document").response
    keywords = res.split('\n')
    # st.session_state.keywords = keywords



# if "selected_keywords" not in st.session_state:
#     selected_keywords

    selected_keywords = st.multiselect('Select topics for questions',keywords, default=keywords[1])

if st.button("Start learning Session"):
    current_keyword = selected_keywords.pop(0)
    question, _ = tutor.generate_question_answer(current_keyword)
    st.write("Question: ", question)
    st.write("Provide your answer and press 'Submit Answer' when ready.")
else:
    st.write("Please select at least one topic.")







answer = st.text_input("Your answer:")
if st.button("Submit Answer"):
    feedback, score = tutor.give_feedback(answer)
    st.write("Feedback: ", feedback)

    if score < tutor.score_threshold:  # if answer is incorrect or partially correct
        # generate subtopic from current_keyword and add it to selected_keywords
        subtopic = tutor.extract_keywords(feedback)  # This should ideally be a more sophisticated subtopic generation, but we'll use keyword extraction for simplicity.
        selected_keywords.insert(0, subtopic[0])  # Insert the first keyword as a subtopic
    elif selected_keywords:  # if there are still selected_keywords left
        selected_keywords.pop(0)  # remove the current keyword

    if selected_keywords:
        current_keyword = selected_keywords[1]
        question, _ = tutor.generate_question_answer(current_keyword)
        st.write("Next question: ", question)
    else:
        st.write("You have completed all the selected topics. Well done!")

# Display chat history on sidebar
st.sidebar.header("Chat History")
for message in tutor._chat_history:
    if message.role == 'system':
        st.sidebar.markdown(f"**System**: {message.content}")
    else:
        st.sidebar.markdown(f"**You**: {message.content}")


# except (AttributeError):
#     print("Error")