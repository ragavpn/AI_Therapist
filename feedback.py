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


feedbackfile = 'feedback.txt'

if os.path.exists(feedbackfile):
    with open(feedbackfile, 'r+') as file:
        feedback = file.read()

needed_text = feedback.split("\n")
input_text=needed_text[0]
output_text=needed_text[1]
feed=needed_text[2]
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

prediction = classifier(input_text)
prediction = prediction[0]

# Sort the predictions by score in descending order
sorted_predictions = sorted(prediction, key=lambda x: x['score'], reverse=True)

# Find the label 
replacwith1 = sorted_predictions[2]['label']
dialogue_emotion1 = sorted_predictions[1]['label']
emotion1 = sorted_predictions[0]['label']

prediction = classifier(output_text)
prediction = prediction[0]

# Sort the predictions by score in descending order
sorted_predictions = sorted(prediction, key=lambda x: x['score'], reverse=True)

# Find the label 
replacwith2 = sorted_predictions[2]['label']
dialogue_emotion2 = sorted_predictions[1]['label']
emotion2 = sorted_predictions[0]['label']



if feed== '4':
    incentive = (f'"{input_text}" - this shows {emotion1} and from past conversation, it also shows {dialogue_emotion1}. '
            f'Due to these emotions being present, an empathetic response is crucial to convey understanding and support. '
            f'Therefore, the two necessary empathetic emotions in a response would be {emotion2} and {dialogue_emotion2}. '
            f'The response to the above statement was "{output_text}". However, this response was not empathetic enough. Make response include {replacwith2} and influence the response with a weight of 30 percent with {emotion2} at 50 percent and {dialogue_emotion2} at 20 percent.')
elif feed== '3':
    incentive = (f'"{input_text}" - this shows {emotion1} and from past conversation, it also shows {dialogue_emotion1}. '
            f'Due to these emotions being present, an empathetic response is crucial to convey understanding and support. '
            f'Therefore, the two necessary empathetic emotions in a response would be {emotion2} and {dialogue_emotion2}. '
            f'The response to the above statement was "{output_text}". However, this response was not empathetic enough. Make response include {replacwith2} and influence the response with a weight of 50 percent with {emotion2} at 30 percent and {dialogue_emotion2} at 20 percent.')
elif feed== '2':
    incentive = (f'"{input_text}" - this shows {emotion1} and from past conversation, it also shows {dialogue_emotion1}. '
            f'Due to these emotions being present, an empathetic response is crucial to convey understanding and support. '
            f'Therefore, the two necessary empathetic emotions in a response would be {emotion2} and {dialogue_emotion2}. '
            f'The response to the above statement was "{output_text}". However, this response was not empathetic enough. Make response include {replacwith2} and influence the response with a weight of 70 percent with {emotion2} at 20 percent and {dialogue_emotion2} at 10 percent.')
elif feed== '1':
    incentive = (f'"{input_text}" - this shows {emotion1} and from past conversation, it also shows {dialogue_emotion1}. '
            f'Due to these emotions being present, an empathetic response is crucial to convey understanding and support. '
            f'Therefore, the two necessary empathetic emotions in a response would be {emotion2} and {dialogue_emotion2}. '
            f'The response to the above statement was "{output_text}". However, this response was not empathetic enough. Make response include {replacwith2} and influence the response with a weight of 80 percent with {emotion2} at 15 percent and {dialogue_emotion2} at 5 percent.')

learning_rate = 0.001

# Prepare the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)

for i in range(3):
    optimizer.zero_grad()
    outputs = model(input_ids=incentive, labels=incentive)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()

# Save the model's weights and biases
model.save_pretrained("gpt2_empathy_model")

# Save the model configuration
model_config = model.config
model_config.save_pretrained("gpt2_empathy_model_config")




