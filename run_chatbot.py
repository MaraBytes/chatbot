import numpy as np
from train_chatbot import train_chatbot
from preprocess_data import load_data, preprocess_data
from sklearn.metrics.pairwise import cosine_similarity
def get_response(user_input, vectors, vectorizer, answers, threshold=0.2):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, vectors).flatten()
    index_of_best_match = np.argmax(similarity)
    
    if similarity[index_of_best_match] < threshold:
        # Fallback response for unknown queries
        return "I'm sorry, I don't understand that. Please ask me something else."
    
    return answers[index_of_best_match]

if __name__ == "__main__":
    data = load_data('dataset.json')
    questions, answers = preprocess_data(data)
    vectors, vectorizer, trained_answers = train_chatbot(questions, answers)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = get_response(user_input, vectors, vectorizer, trained_answers)
        print("Chatbot:", response)
