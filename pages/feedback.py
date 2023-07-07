import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import difflib

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
stemmer = PorterStemmer()

class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history
        self.expected_answer = ''
        self.text = ''

    def reset(self):
        self._chat_history = []

    def generate_question(self) -> str:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a broad question about the following text: {self.text} and provide the expected answer")])
        split_message = message.message.content.split('\n')
        self.expected_answer = split_message[1]
        st.write(split_message[1])
        
        return split_message[0]

    # Initialize the stemmer

    def get_score(self, user_answer: str) -> float:
        # use self.expected_answer directly in this method
        sequence_matcher = difflib.SequenceMatcher(None, self.expected_answer, user_answer)
        score = sequence_matcher.ratio()
        return score

    def compare_answers(self, user_answer: str) -> float:
        return self.get_score(self.expected_answer, user_answer)

    def give_feedback(self, answer: str) -> str:
        self._chat_history.append(ChatMessage(role="user", content=answer))

        feedback_instructions = """
        Please provide detailed feedback based on the following principles:
        (1) Help clarify what good performance is (goals, criteria, expected standards)
        (2) Facilitate the development of self-assessment (reflection) in learning
        (3) Deliver high quality information to students about their learning
        (4) Encourage teacher and peer dialogue around learning
        (5) Encourage positive motivational beliefs and self-esteem
        (6) Provide opportunities to close the gap between current and desired performance
        (7) Provide information to teachers that can be used to help shape teaching

        Do not give away the correct answer if the answer is incorrect or only partly correct.
        Make sure to mention the principle numbers that are relevant to your feedback.
        If the answer is incorrect or only partly correct, guide the user to try again.

        Expected answer: {self.expected_answer}
        User's answer: {answer}
        """

        self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

        message = self._llm.chat(self._chat_history)
        return message.message.content


tutor = TutorAgent()

st.title("AI Tutor")
tutor.text = st.text_area("Input text for learning:", "Enter text here...")

if st.button("Start learning session"):
    question = tutor.generate_question()
    st.write("Question: ", question)
    st.write("Provide your answer and press 'Submit Answer' when ready.")

answer = st.text_input("Your answer:")
if st.button("Submit Answer"):
    feedback = tutor.give_feedback(answer)
    st.write("Feedback: ", feedback)

# Display chat history on sidebar
st.sidebar.header("Chat History")
for message in tutor._chat_history:
    if message.role == 'system':
        st.sidebar.markdown(f"**System**: question")
    else:
        st.sidebar.markdown(f"**You**: {message.content}")
