from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_data import load_data, preprocess_data

def train_chatbot(questions, answers):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    return vectors, vectorizer, answers

if __name__ == "__main__":
    data = load_data('dataset.json')
    questions, answers = preprocess_data(data)
    vectors, vectorizer, trained_answers = train_chatbot(questions, answers)
