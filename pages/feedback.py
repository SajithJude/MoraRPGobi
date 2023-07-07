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

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history
        self.score_threshold = 0.7  # adjust this as per your requirements

    def reset(self):
        self._chat_history = []

    def extract_keywords(self, text: str) -> List[str]:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Please list 10 keywords or topics from the following text: {text}")])
        keywords = message.message.content.split('\n')  # Assuming the model returns a newline-separated list
        return keywords

    def generate_question_answer(self, keyword: str) -> (str, str):
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a question about the topic: {keyword} and also provide the expected answer.")])
        question, expected_answer = message.message.content.split('\n')  # Assuming the model returns question and answer separated by a newline
        self.expected_answer = expected_answer
        return question, expected_answer

    def give_feedback(self, user_answer: str) -> (str, float):
        self._chat_history.append(ChatMessage(role="user", content=user_answer))

        feedback_instructions = """
        // ...
        """
        
        self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

        message = self._llm.chat(self._chat_history)
        feedback = message.message.content

        score = self.get_score(user_answer, self.expected_answer)
        return feedback, score

    def get_score(self, user_answer: str, expected_answer: str) -> float:
        return sentence_bleu([expected_answer.split()], user_answer.split())


tutor = TutorAgent()

st.title("AI Tutor")
text = st.text_area("Input text for learning:", "Enter text here...")

if st.button("Start learning session"):
    keywords = tutor.extract_keywords(text)
    selected_keywords = st.multiselect('Select topics for questions', keywords)

    if selected_keywords:
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
        subtopic = tutor.extract_keywords(current_keyword)  # This should ideally be a more sophisticated subtopic generation, but we'll use keyword extraction for simplicity.
        selected_keywords.insert(0, subtopic[0])  # Insert the first keyword as a subtopic
    elif selected_keywords:  # if there are still selected_keywords left
        selected_keywords.pop(0)  # remove the current keyword

    if selected_keywords:
        current_keyword = selected_keywords[0]
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
