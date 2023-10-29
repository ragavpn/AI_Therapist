# GRASSH_ML

# Netcom Problem Statement:
# Emotion-Detecting Therapeutic Chat-bot

We are using emotion detection in chat to create a more robust and interactive chat experience for the user, with the underlying goal of acting essentially as an AI therapist. 


The model uses an emotion classifier to classify input text emotion (by using a Text Transformer)
Then, we use prompt engineering (chain-of-thought prompting) to fine tune gpt-2 to the task of offering therapeutic advice with reference to an emotion classified therapy transcript database. 
The response is taking in classified emotions at different weights. These weights are changed on the basis of feedback from the user after recieving the responses.
This ensures that the model gives the user a personalised experience on two parameters: 
1. User feedback of their satisfaction with the responses
2. User emotion detected by text transformer

![image](https://github.com/Medici357/Grassh_shit/assets/127466814/c96f0269-9db2-4567-8604-b2d55fc55736)

# Usage
Download the bin file from https://drive.google.com/file/d/1q94PkFwAPM0_IWhn_D3I-sa4QaH3bVwN/view?usp=drive_link and paste it in the "gpt2_empathy_model" for the code to run.
