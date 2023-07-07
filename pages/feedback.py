import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List

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
        self.expected_answer = split_message[1] if len(split_message) > 1 else ''
        return split_message[0]

    # Initialize the stemmer

    def get_score(expected_answer: str, user_answer: str) -> float:
        # Tokenize and stem the words in each sentence
        expected_words = set(stemmer.stem(word) for word in word_tokenize(expected_answer))
        user_words = set(stemmer.stem(word) for word in word_tokenize(user_answer))

        # Calculate the score as the size of the intersection divided by the size of the union
        score = len(expected_words & user_words) / len(expected_words | user_words)

        return score

    def compare_answers(self, user_answer: str) -> float:
        return self.get_score(self.expected_answer, user_answer)

    def give_feedback(self, answer: str) -> str:
        score = self.compare_answers(answer)
        self._chat_history.append(ChatMessage(role="user", content=answer))
        feedback = ''

        if score < 0.8:  # Threshold to consider the answer as correct or not.
            feedback = "Your answer is not quite correct. Try again."
            # Generate a new question based on the same text.
            question = self.generate_question()
            self._chat_history.append(ChatMessage(role="system", content=f"New question: {question}"))
        else:
            feedback = "Good job! Your answer is correct."

        self._chat_history.append(ChatMessage(role="system", content=feedback))
        return feedback


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
        st.sidebar.markdown(f"**System**: {message.content}")
    else:
        st.sidebar.markdown(f"**You**: {message.content}")
