import joblib

def load_models():
    vectorizer = joblib.load('vectorizer_model.joblib')
    classifier = joblib.load('classifier_model.joblib')
    return vectorizer, classifier

def chat_with_bot(vectorizer, classifier):
    print("Chatbot: Hi! Ask me anything or type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

       
        user_input_vectorized = vectorizer.transform([user_input])

        predicted_answer = classifier.predict(user_input_vectorized)

        print("Chatbot:", predicted_answer[0])

if __name__ == "__main__":
    vectorizer, classifier = load_models()

    
    chat_with_bot(vectorizer, classifier)
