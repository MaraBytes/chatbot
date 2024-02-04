import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QSizePolicy
from PyQt5.QtGui import QTextCursor
import numpy as np
from train_chatbot import train_chatbot
from preprocess_data import load_data, preprocess_data
from sklearn.metrics.pairwise import cosine_similarity

class ChatBotApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load and preprocess data
        data = load_data('dataset.json')
        questions, answers = preprocess_data(data)
        self.vectors, self.vectorizer, self.trained_answers = train_chatbot(questions, answers)

        # UI components
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your question here...")
        self.send_button = QPushButton("Send")
        self.clear_button = QPushButton("Clear Conversation")
        self.quit_button = QPushButton("Quit")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.chat_display)
        layout.addWidget(self.input_box)
        layout.addWidget(self.send_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.quit_button)

        self.setLayout(layout)

        # Connect signals
        self.send_button.clicked.connect(self.process_input)
        self.clear_button.clicked.connect(self.clear_conversation)
        self.quit_button.clicked.connect(self.close)

        self.setWindowTitle("ChatBot")
        self.setGeometry(100, 100, 600, 400)

    def append_to_chat(self, text, is_user=False):
        if is_user:
            self.chat_display.setTextColor("blue")
        else:
            self.chat_display.setTextColor("green")
        self.chat_display.append(text)
        self.chat_display.moveCursor(QTextCursor.End)

    def process_input(self):
        user_input = self.input_box.text()
        self.append_to_chat("You: " + user_input, is_user=True)

        if user_input.lower() == 'exit':
            self.append_to_chat("Chatbot: Goodbye!")
            sys.exit()

        response = self.get_response(user_input)
        self.append_to_chat("Chatbot: " + response)

        # Clear input box
        self.input_box.clear()

    def get_response(self, user_input, threshold=0.2):
        user_vector = self.vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vector, self.vectors).flatten()
        index_of_best_match = np.argmax(similarity)

        if similarity[index_of_best_match] < threshold:
            # Fallback response for unknown queries
            return "I'm sorry, I don't understand that. Please ask me something else."

        return self.trained_answers[index_of_best_match]

    def clear_conversation(self):
        self.chat_display.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    chat_app = ChatBotApp()
    chat_app.show()
    sys.exit(app.exec_())
