with open('QA_Beauty.json', 'r', encoding='utf-8') as file:
    json_content = file.read()

# Replace single quotes with double quotes
json_content = json_content.replace("'", "\"")

# Save the modified content to a new text file
with open('preprocessed_json.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(json_content)

print("Preprocessing completed. Data saved to 'preprocessed_json.txt'")
