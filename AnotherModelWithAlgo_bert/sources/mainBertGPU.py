import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

# Check if GPU is available
device = torch.device("cuda")

#use GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Load the DNS traffic dataset and preprocess the data
# Assuming you have a parquet file named 'dns_traffic.parquet' in the current directory
# Modify the code below to load and preprocess the data accordingly
#df = pd.read_parquet('dns_traffic.parquet')
df = pd.read_pickle('dns_parquet_files.pkl')
#df = pd.read_csv('dns_totalPack.csv')


#df = pd.read_csv('D:\\DeepLearning DNS Akamai\\ariel_university_axLogs\\separate\\dns_totalPack.csv')

# Extract relevant features from the parquet file
request_ts = df['request_ts'].values
response_ts = df['response_ts'].values
client_token_dec = df['client_token_dec'].values
qname = df['qname'].values
qtype = df['qtype'].values
response_flags = df['response_flags'].values
answer = df['answer'].values
nx_domains = df['nx_domains'].values
l1_cache_hit = df['l1_cache_hit'].values
response_size = df['response_size'].values
category_id = df['category_id'].values
in_BL = df['in_BL'].values


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode the relevant features
encoded_inputs = tokenizer(request_ts, response_ts, client_token_dec, qname, qtype, response_flags, answer, nx_domains, l1_cache_hit, response_size, category_id, in_BL, return_tensors='pt', padding=True, truncation=True)

# Create a TensorDataset
dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['token_type_ids'], encoded_inputs['attention_mask'], torch.tensor(in_BL))

# Split the dataset into training and testing sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create a dataloader for the training set
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a dataloader for the testing set
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Set the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Set the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
model.train()
for epoch in range(10):
    for batch in train_dataloader:
        # Get the inputs
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = batch[3].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels).to(device)

        # Compute the loss
        loss = criterion(outputs[1], labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

    print(f'Epoch {epoch+1} loss: {loss.item()}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_dataloader:
        # Get the inputs
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = batch[3].to(device)

        # Forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).to(device)

        # Get the predictions
        _, predicted = torch.max(outputs[0], 1)

        # Update the counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total}')

# Save the model
torch.save(model.state_dict(), 'model.pt')




# Assuming you have NumPy arrays for each feature in the dataset: 'Request_ts', 'Response_ts', 'client_token_dec', 'qname', 'qtype', 'qclass', 'response_flags', 'answer', 'Nx_domains', 'L1_cache_hit', 'response_size', 'in_BL'
# Modify the code below to load and preprocess the data accordingly

# Combine relevant features into a single textual feature
X = [f"request_ts: {ts} response_ts: {rts} client_token: {client_token} Qname: {qname} Qtype: {qtype}  Response_flags: {response_flags} Answer: {answer} nx_domains: {nx_domains} l1_cache_hit: {l1_cache_hit} Response_size: {response_size} category_id: {category_id}" for ts, rts, client_token, qname, qtype, response_flags, answer, nx_domains, l1_cache_hit, response_size, category_id ,in_BL in zip(request_ts, response_ts, client_token_dec, qname, qtype, response_flags, answer, nx_domains, l1_cache_hit, response_size, category_id,in_BL)]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, in_BL, test_size=0.2, random_state=42)


# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = np.max(y_train_encoded) + 1
y_train_tensor = torch.tensor(y_train_encoded)
y_test_tensor = torch.tensor(y_test_encoded)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model.to(device)

# Tokenize the input sequences
X_train_tokens = tokenizer.batch_encode_plus(
    X_train,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
X_test_tokens = tokenizer.batch_encode_plus(
    X_test,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Convert tokenized inputs to tensors
X_train_input_ids = X_train_tokens['input_ids'].to(device)
X_train_attention_mask = X_train_tokens['attention_mask'].to(device)
X_test_input_ids = X_test_tokens['input_ids'].to(device)
X_test_attention_mask = X_test_tokens['attention_mask'].to(device)

# Move the model to the GPU
model.to(device)


# Create PyTorch datasets
train_dataset = TensorDataset(X_train_input_ids, X_train_attention_mask, y_train_tensor)
test_dataset = TensorDataset(X_test_input_ids, X_test_attention_mask, y_test_tensor)

# Create PyTorch data loaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(5):
    total_loss = 0
    for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
        # Move inputs and labels to the GPU
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1} - Average Training Loss: {average_loss:.4f}')

# Evaluation
model.eval()
total_loss = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch_input_ids, batch_attention_mask, batch_labels in test_dataloader:
        # Move inputs and labels to the GPU
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        logits = outputs.logits

        # Calculate predictions and accuracy
        _, predictions = torch.max(logits, dim=1)
        correct_predictions += torch.sum(predictions == batch_labels).item()
        total_predictions += len(batch_labels)

        total_loss += loss.item()

average_loss = total_loss / len(test_dataloader)
accuracy = correct_predictions / total_predictions

print(f'Test Loss: {average_loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


