import torch
import time
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
import os, warnings
from torch.utils.data import DataLoader, TensorDataset
CUDA_LAUNCH_BLOCKING=1
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_data(file_path, file_path_val):
    training_data = []
    validation_data = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            if lines[i].strip():  # Skip empty lines
                combined_line = "".join(lines[i:i+3]).replace("\n", "")
                training_data.append(combined_line)

    with open(file_path_val, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            if lines[i].strip():  # Skip empty lines
                combined_line = "".join(lines[i:i+3]).replace("\n", "")
                validation_data.append(combined_line)
    
    return training_data, validation_data

def Data_Loader(training_data, validation_data):
    # Tokenize the data
    input_ids = [tokenizer.encode(text) for text in training_data]
    val_input_ids = [tokenizer.encode(text) for text in validation_data]

    # Convert lists to tensors and move to device
    input_ids = [torch.tensor(ids).to(device) for ids in input_ids]
    val_input_ids = [torch.tensor(ids).to(device) for ids in val_input_ids]

    # Pad your sequences
    input_ids = pad_sequence(input_ids, batch_first=True)
    val_input_ids = pad_sequence(val_input_ids, batch_first=True)

    # Convert input_ids to TensorDataset
    train_dataset = TensorDataset(input_ids)
    val_dataset = TensorDataset(val_input_ids)

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, input_ids, val_input_ids

def get_model(checkpoint_path):

    # Load or initialize model, optimizer, scheduler
    if os.path.exists(checkpoint_path):
        # Load checkpoint if it exists
        checkpoint = torch.load(checkpoint_path)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(input_ids) * epochs)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded saved model")
    else:
        # Initialize model, optimizer, scheduler if checkpoint doesn't exist
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(input_ids) * epochs)

    return model, optimizer, scheduler

def save_model(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)

    print("Model Saved")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "data/notes_train.txt"
file_path_val = "data/notes_val.txt"

training_data, validation_data = get_data(file_path, file_path_val)

# Hyperparameters
model_name = "gpt2"
checkpoint_path = "gpt2_empathy_checkpoint.pt"
learning_rate = 0.001
epochs = 5
batch_size = 4
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


train_dataloader, val_dataloader, input_ids, val_input_ids = Data_Loader(training_data, validation_data)

model, optimizer, scheduler = get_model(checkpoint_path)

# Set the pad_token_id to indicate the end of generated text
model.config.pad_token_id = model.config.eos_token_id

# Fine-tune the model
for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()

    model.train()
    batch_count = 0
    for batch in train_dataloader:
        try:
            optimizer.zero_grad()
            outputs = model(input_ids=batch[0], labels=batch[0])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_count}/{len(train_dataloader)} - Training Loss: {loss.item():.4f}")

        except KeyboardInterrupt:
            print("Force Checkpoint")
            save_model(model, optimizer, scheduler, epoch, loss, checkpoint_path)
    
    save_model(model, optimizer, scheduler, epoch, loss, checkpoint_path)
    print("Model Saved")

    model.eval()
    val_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(input_ids=batch[0], labels=batch[0])
            val_loss += outputs.loss.item()
            batch_count += 1
            print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_count}/{len(val_dataloader)} - Validation Loss: {outputs.loss.item():.4f}")

    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {total_loss/len(train_dataloader):.4f} - Validation Loss: {val_loss/len(val_dataloader):.4f} - Time: {epoch_time:.2f} seconds")