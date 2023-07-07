import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history
        self.expected_answer = ''

    def reset(self):
        self._chat_history = []

    def generate_question(self, text: str) -> str:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a broad question about the following text: {text} and provide the expected answer")])
        split_message = message.message.content.split('\n')
        self.expected_answer = split_message[1] if len(split_message) > 1 else ''
        return split_message[0]

    def compare_answers(self, user_answer: str) -> str:
        message = self._llm.chat([ChatMessage(role="system", content=f"Compare the following answers:\nExpected: {self.expected_answer}\nUser: {user_answer}")])
        return message.message.content

    def give_feedback(self, answer: str) -> str:
        comparison = self.compare_answers(answer)
        self._chat_history.append(ChatMessage(role="user", content=answer))
        self._chat_history.append(ChatMessage(role="system", content=comparison))

        feedback_instructions = """
        Please provide feedback based on the following principles...
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
