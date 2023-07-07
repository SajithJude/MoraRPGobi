import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
class TutorAgent(YourOpenAIAgent):
    def __init__(self, tools: Sequence[BaseTool] = [], llm: OpenAI = OpenAI(model="gpt-3.5-turbo-0613"), chat_history: List[ChatMessage] = []):
        super().__init__(tools, llm, chat_history)
        self.expected_answer = ""
        self.threshold_score = 0.7  # Set according to your requirements

    def generate_question_and_answer(self, prompt: str):
        self._chat_history.append(ChatMessage(role="system", content=prompt))
        question_message = self._llm.chat(self._chat_history)
        self._chat_history.append(question_message.message)
        
        self._chat_history.append(ChatMessage(role="system", content="Please provide an answer to the question."))
        answer_message = self._llm.chat(self._chat_history)
        self._chat_history.append(answer_message.message)

        self.expected_answer = answer_message.message.content
        return question_message.message.content

    def evaluate_answer(self, user_answer: str) -> float:
        # Use the OpenAI API to evaluate the similarity between the user's answer and the expected answer
        prompt = f"Considering the context and semantics, rate the similarity between the following two sentences on a scale of 0 to 1:\n\nSentence 1: {self.expected_answer}\nSentence 2: {user_answer}"
        self._chat_history.append(ChatMessage(role="system", content=prompt))
        rating_message = self._llm.chat(self._chat_history)
        self._chat_history.append(rating_message.message)
        
        score = float(rating_message.message.content)
        return score

    def give_feedback_and_followup_question(self, user_answer: str):
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

        feedback_and_question_message = self._llm.chat(self._chat_history)
        return feedback_and_question_message.message.content


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
