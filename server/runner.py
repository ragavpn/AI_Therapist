from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import warnings

CUDA_LAUNCH_BLOCKING = 1
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cpu")

# Classifier pipeline for text classification (DistilBERT)
classifier = pipeline("text-classification", model="bdotloh/distilbert-base-uncased-empathetic-dialogues-context", return_all_scores=True)

# Load the pre-trained RoBERTa model and tokenizer
model_name = "gpt2"
# saved_model_dir = "gpt2_empathy_model"
checkpoint_path = "gpt2_empathy_checkpoint.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded checkpoint")
else:
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad_token_id to indicate the end of generated text
model.config.pad_token_id = model.config.eos_token_id


text_path = 'text.txt'

if os.path.exists(text_path):
    with open(text_path, 'r') as file:
        text = file.read()


prediction = classifier(text)
prediction = prediction[0]

# Sort the predictions by score in descending order
sorted_predictions = sorted(prediction, key=lambda x: x['score'], reverse=True)

# Find the label with the second-highest score
dialogue_emotion = sorted_predictions[1]['label']
emotion = sorted_predictions[0]['label']
prompt = f"User Input: {text}\nThis shows the emotion {emotion} and from past conversation, it also shows the emotion {dialogue_emotion} currently."

# Generate text based on the prompt using RoBERTa
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_text = generated_text.find("The response to the above statement can be - ")
generated_text = generated_text[generated_text + len("The response to the above statement can be - ")+1:]
generated_text = generated_text[:generated_text.find("!!")-2]

if os.path.exists(text_path):
    with open(text_path, 'w') as file:
        file.write(generated_text)
