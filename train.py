from transformers import pipeline
import torch
from transformers import RobertaForCausalLM, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import random
import os, warnings
CUDA_LAUNCH_BLOCKING=1
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the CSV file
csv_file = "train.csv"  # Replace with your CSV file's path
df = pd.read_csv(csv_file)

# Check if a CUDA-compatible GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer
classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)

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
        response = (
            f'"{input_text}" - this shows {emotion1} and from past conversation, it also shows {dialogue_emotion1}. '
            f'Due to these emotions being present, an empathetic response is crucial to convey understanding and support. '
            f'Therefore, the two necessary empathetic emotions in a response would be {emotion2} and {dialogue_emotion2}. '
            f'The response to the above statement can be "{output_text}".'
        )

        training_data.append(response)

training_data = training_data[3000:4000]  # Limit the training data to 100 examples
    
# Load the pre-trained gpt2 model and tokenizer
model_name = "gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Check if a saved model exists
saved_model_dir = "gpt2_empathy_model"
if os.path.exists(saved_model_dir):
    # Load the saved model configuration and weights
    model = GPT2LMHeadModel.from_pretrained(saved_model_dir).to(device)
    print("Loaded saved model")
else:
    # Load the model from Hugging Face without fine-tuning
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Tokenize the training data
input_ids = [tokenizer.encode(text, return_tensors="pt").to(device) for text in training_data]

# Set the pad_token_id to indicate the end of generated text
model.config.pad_token_id = model.config.eos_token_id

# Define hyperparameters for fine-tuning
learning_rate = 0.001
epochs = 50

# Prepare the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(input_ids) * epochs)

# Fine-tune the model
for epoch in range(epochs):
    random.shuffle(input_ids)  # Shuffle the training data for better training
    total_loss = 0
    for batch in input_ids:
        optimizer.zero_grad()
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    average_loss = total_loss / len(input_ids)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")


text = input("Enter text: ")

prediction = classifier(text)
prediction = prediction[0]

# Sort the predictions by score in descending order
sorted_predictions = sorted(prediction, key=lambda x: x['score'], reverse=True)

# Find the label with the second-highest score
dialogue_emotion = sorted_predictions[1]['label']
emotion = sorted_predictions[0]['label']

# Input prompt
prompt = "{} - this shows {} and from past conversation, it also shows {}.".format(text, emotion, dialogue_emotion)

# Generate text based on the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
needed_text = generated_text.split("The response to the above statement can be")[-1].strip()[1:]
flag = needed_text.find("\"")
needed_text = needed_text[:flag]
print("Generated Text: ", needed_text)

# Pass the substring that follows "The response to the above statement can be" through the classifier
substring_to_classify = generated_text.split("The response to the above statement can be")[-1].strip()
prediction = classifier(substring_to_classify)
print("Emotion Prediction:", prediction)


# Save the model configuration
model_config = model.config
model_config.save_pretrained("gpt2_empathy_model_config")

# Save the model's weights and biases
model.save_pretrained("gpt2_empathy_model")