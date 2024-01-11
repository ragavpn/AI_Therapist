import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
import os
import time
logging.basicConfig(level=logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = []
validation_data = []

file_path = "data/notes_train.txt"

with open(file_path, "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 4):
        if lines[i].strip():
            combined_line = "".join(lines[i:i + 3]).replace("\n", "")
            currently = combined_line.find("currently. ")
            training_data.append([combined_line[:currently + 11], combined_line[currently + 12:]])

file_path_val = "data/notes_val.txt"

with open(file_path_val, "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 4):
        if lines[i].strip():
            combined_line = "".join(lines[i:i + 3]).replace("\n", "")
            currently = combined_line.find("currently. ")
            validation_data.append([combined_line[:currently + 11], combined_line[currently + 12:]])

model_name = "roberta-base"
saved_model_dir = "roberta_empathy_model"
LEARNING_RATE = 1e-05
EPOCHS = 1
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
MAX_LEN = 256

tokenizer = RobertaTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = [i[0] for i in dataframe]
        self.targets = [i[1] for i in dataframe]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        target = self.targets[index]

        # Tokenize the text and target
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        targets = self.tokenizer.encode(
            target,
            add_special_tokens=False,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=False,
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }

training_set = SentimentData(training_data, tokenizer, MAX_LEN)
validation_set = SentimentData(validation_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 256)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    total_loss = 0
    start_time = time.time()

    model.train()
    batch_count = 0
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float64)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        total_loss += loss.item()
        batch_count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"   Epoch {epoch + 1}/{EPOCHS} - Batch {batch_count}/{len(training_loader)} - Training Loss: {loss.item():.4f}")

        

    # Validation
    model.eval()
    val_loss = 0
    batch_count = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float64)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            batch_count += 1

            print(f"   Epoch {epoch + 1}/{EPOCHS} Batch {batch_count}/{len(training_loader)} - Validation Loss: {loss.item():.4f}")

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"   Epoch {epoch + 1}/{EPOCHS} - Training Loss: {total_loss/len(training_loader):.4f} - Validation Loss: {val_loss/len(validation_loader):.4f} - Time: {epoch_time:.2f} seconds")



output_model_file = f'{saved_model_dir}/roberta_empathy_model.bin'

# Save the model
torch.save(model, output_model_file)
tokenizer.save_vocabulary(saved_model_dir)
optimizer.save_pretrained(saved_model_dir)

