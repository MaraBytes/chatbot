import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get('data', [])

def preprocess_data(data):
    questions = [entry['question'] for entry in data]
    answers = [entry['answer'] for entry in data]
    return questions, answers
