from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from preprocess_data import load_data, preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def train_chatbot(questions, answers):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    
    classifier = LinearSVC()
    classifier.fit(vectors, answers)

    return vectorizer, classifier

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('dataset.json')
    questions, answers = preprocess_data(data)

    # Train the chatbot
    vectorizer, classifier = train_chatbot(questions, answers)

    # Save the trained models to use later
    joblib.dump(vectorizer, 'vectorizer_model.joblib')
    joblib.dump(classifier, 'classifier_model.joblib')
