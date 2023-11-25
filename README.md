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

# Initialization
Download the rep and run "pip install -r requirements.txt". The run "git lfs install" followed by "git lfs pull". Then go to the client folder and run "npm install" (if there is an error saying react-scripts not found when running the client run "npm install react-scripts").

# Running
Open two terminals. Go to the client folder in the first terminal and run "npm run start". Go to the server folder in the second terminal and run "python -m flask run". 

# Future Implementations

<img width="934" alt="image-20221016141015236" src="https://github.com/ragavpn/GRASSH_ML/assets/118587215/54f1bbf1-0e2d-4097-85de-71bba855531b">

With the current hardware possessed by us, we are not able to train the ML Model that we intend to implemment. However, we will implement this soon getting the said required hardware.
