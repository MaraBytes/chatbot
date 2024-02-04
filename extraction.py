import re

with open('preprocessed_json.txt', 'r') as file:
    data = file.read()


pattern = re.compile(r'"questionText": u"(.*?)",.*?"answerText": u"(.*?)"')

matches = pattern.findall(data)

extracted_data = [(match[0], match[1]) for match in matches]

with open('extracted_data.txt', 'w') as output_file:
    for question, answer in extracted_data:
        output_file.write(f'Question: {question}\nAnswer: {answer}\n\n')
