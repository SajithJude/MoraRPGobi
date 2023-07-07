import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history

    def reset(self):
        self._chat_history = []

    def generate_question(self, text: str) -> str:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a broad question about the following text: {text}")])
        return message.message.content

    def give_feedback(self, answer: str) -> str:
        self._chat_history.append(ChatMessage(role="user", content=answer))

        feedback_instructions = """
        Please provide feedback based on the following principles:
        (1) Help clarify what good performance is (goals, criteria, expected standards)
        (2) Facilitate the development of self-assessment (reflection) in learning
        (3) Deliver high quality information to students about their learning
        (4) Encourage teacher and peer dialogue around learning
        (5) Encourage positive motivational beliefs and self-esteem
        (6) Provide opportunities to close the gap between current and desired performance
        (7) Provide information to teachers that can be used to help shape teaching
        Remember to:
        - Not give away the correct answer if the answer is incorrect or only partly correct.
        - Mention the principle numbers used while giving feedback.
        - Say 'try again' if the answer is incorrect or only partly correct.
        """
        
        self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

        message = self._llm.chat(self._chat_history)
        return message.message.content


tutor = TutorAgent()

st.title("AI Tutor")
text = st.text_area("Input text for learning:", "Enter text here...")

if st.button("Start learning session"):
    question = tutor.generate_question(text)
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
        st.sidebar.markdown(f"**System**: {message.content}")
    else:
        st.sidebar.markdown(f"**You**: {message.content}")
