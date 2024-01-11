import pandas as pd
import jsonlines
import os

def format_text(input_text, emotion1, dialogue_emotion1, output_text, emotion2, dialogue_emotion2):
    # Generate an empathetic response
    response = (
        f'User Input - "{input_text}"\nThis shows the emotion "{emotion1}" and from past conversation, it also shows the emotion "{dialogue_emotion1}" currently. '
        f'Due to these emotions being present, an empathetic response should contain the emotions "{emotion2}" and "{dialogue_emotion2}".\n'
        f'The response to the above statement can be - "{output_text}".\n'
    )
    return response


# Read the CSV file
csv_file = "train.csv"  
df = pd.read_csv(csv_file)

# Initialize variables to store conversation context
input_text = ""
emotion1 = ""
dialogue_emotion1 = ""

# Provide some example data for fine-tuning
training_data = []

# Loop through the CSV rows
for index, row in df.iterrows():
    if index % 2 == 0:
        # This row contains "input" and related emotions
        input_text = row["text"]
        emotion1 = row["emotion"]
        dialogue_emotion1 = row["dialogue_emotion"]
    else:
        # This row contains "output" and related emotions
        output_text = row["text"]
        emotion2 = row["emotion"]
        dialogue_emotion2 = row["dialogue_emotion"]
        # Generate an empathetic response
        response = format_text(input_text, emotion1, dialogue_emotion1, output_text, emotion2, dialogue_emotion2)
        training_data.append(response)

csv_file = "test.csv"  
df = pd.read_csv(csv_file)
val_data = []

# Loop through the CSV rows
for index, row in df.iterrows():
    if index % 2 == 0:
        # This row contains "input" and related emotions
        input_text = row["text"]
        emotion1 = row["emotion"]
        dialogue_emotion1 = row["dialogue_emotion"]
    else:
        # This row contains "output" and related emotions
        output_text = row["text"]
        emotion2 = row["emotion"]
        dialogue_emotion2 = row["dialogue_emotion"]
    
        # Generate an empathetic response
        response = format_text(input_text, emotion1, dialogue_emotion1, output_text, emotion2, dialogue_emotion2)
        val_data.append(response)


def write_to_txt(data, file_path):
    # Clear the file before writing
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as writer:
        for item in data:
            writer.write(f'{item}\n')

file_path_train = 'notes_train.txt'
file_path = 'notes_val.txt'
write_to_txt(training_data, file_path_train)
write_to_txt(val_data, file_path)