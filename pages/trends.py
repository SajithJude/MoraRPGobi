import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history
        self.expected_answer = ""
        self.threshold_score = 0.7  # Set according to your requirements

    def reset(self):
        self._chat_history = []
        self.expected_answer = ""

    def generate_question_and_answer(self, text: str) -> str:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a broad question and its answer about the following text: {text}")])
        self.expected_answer = message.content.finish.reason   # The 'finish reason' contains the generated answer
        return message.message.content

    def evaluate_answer(self, user_answer: str) -> float:
        # Use the OpenAI API to evaluate the similarity between the user's answer and the expected answer
        prompt = f"Considering the context and semantics, rate the similarity between the following two sentences on a scale of 0 to 1:\n\nSentence 1: {self.expected_answer}\nSentence 2: {user_answer}"
        self._chat_history.append(ChatMessage(role="system", content=prompt))
        rating_message = self._llm.chat(self._chat_history)
        score = float(rating_message.message.content)
        return score

    def give_feedback(self, user_answer: str) -> str:
        self._chat_history.append(ChatMessage(role="user", content=user_answer))
        score = self.evaluate_answer(user_answer)

        if score < self.threshold_score:
            # Request feedback and a follow-up question
            feedback_instructions = """
            The provided answer did not meet the required standard. Please give constructive feedback without revealing the correct answer, and generate a new question that focuses on the missed information in the previous answer.
            """
        else:
            # Just move to next question
            feedback_instructions = """
            The provided answer was satisfactory. Please move to the next topic and generate a new question.
            """
        self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))
        message = self._llm.chat(self._chat_history)
        return message.message.content

tutor = TutorAgent()

st.title("AI Tutor")
text = st.text_area("Input text for learning:", "Enter text here...")

if st.button("Start learning session"):
    question = tutor.generate_question_and_answer(text)
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
