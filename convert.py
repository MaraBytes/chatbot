import json

# Function to extract questions and answers from the text file
def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    current_question = None

    for line in lines:
        line = line.strip()

        if line.startswith("Question:"):
            current_question = line[10:].strip()
        elif line.startswith("Answer:"):
            answer = line[8:].strip()
            if current_question:
                data.append({"question": current_question, "answer": answer})
                current_question = None

    return {"data": data}

# Function to save data to a JSON file
def save_to_json(data, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# Specify your input text file path
input_file_path = 'extracted_data.txt'

# Specify the output JSON file path
output_json_file_path = 'dataset.json'

# Extract data from the text file
result_data = extract_data(input_file_path)

# Save the data to a JSON file
save_to_json(result_data, output_json_file_path)

print(f"Data has been extracted and saved to {output_json_file_path}")
