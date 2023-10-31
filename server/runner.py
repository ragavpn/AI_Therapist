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

text_path = 'text.txt'

if os.path.exists(text_path):
    with open(text_path, 'r') as file:
        text = file.read()

print("Read text: ", text)

# Check if a CUDA-compatible GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer
classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
    
# Load the pre-trained gpt2 model and tokenizer
model_name = "gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Check if a saved model exists
saved_model_dir = "gpt2_empathy_model"
model = GPT2LMHeadModel.from_pretrained(saved_model_dir).to(device)


# Set the pad_token_id to indicate the end of generated text
model.config.pad_token_id = model.config.eos_token_id

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

# needed_text=  "hello"

text_path = 'text.txt'

if os.path.exists(text_path):
    with open(text_path, 'w') as file:
        file.write(needed_text)