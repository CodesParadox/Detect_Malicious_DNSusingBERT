import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DNS traffic dataset and preprocess the data
df = pd.read_pickle('dns_parquet_files.pkl')

# Extract relevant fields from the dataset
request_ts = df['request_ts']
response_ts = df['response_ts']
client_token_dec = df['client_token_dec']
qname = df['qname']
qtype = df['qtype']
qclass = df['qclass']
response_flags = df['response_flags']
answer = df['answer']
nx_domains = df['nx_domains']
l1_cache_hit = df['l1_cache_hit']
response_size = df['response_size']
categoty_id = df['category_id']
list_id = df['list_id']
in_BL = df['in_BL']


# Combine relevant features into a single textual feature
X = [f"Request_ts: {ts} Response_ts: {rts} Client_token: {client_token} Qname: {qname} Qtype: {qtype} Qclass: {qclass} Response_flags: {response_flags} Answer: {answer} Nx_domains: {nx_domains} L1_cache_hit: {l1_cache_hit} Response_size: {response_size}" for ts, rts, client_token, qname, qtype, qclass, response_flags, answer, nx_domains, l1_cache_hit, response_size, in_BL in zip(request_ts, response_ts, client_token_dec, qname, qtype, qclass, response_flags, answer, nx_domains, l1_cache_hit, response_size, in_BL)]

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

# Train the BERT model
def train(model, train_dataloader, optimizer, criterion, device, epochs):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_dataloader)}")

# Test the BERT model
def evaluate(model, test_dataloader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs[0], dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Train and test the BERT model
epochs = 5
train(model, train_dataloader, optimizer, criterion, device, epochs)
evaluate(model, test_dataloader, device)

# Make predictions for the DNS queries in the test set
def predict(model, test_dataloader, device):
    model.eval()
    model.to(device)
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs[0], dim=1)
            predictions.extend(predicted.tolist())

    return predictions

# Make predictions for the DNS queries in the test set
predictions = predict(model, test_dataloader, device)

# Perform further analysis or actions based on the predicted labels
# For example, you can print the predicted labels
predicted_labels = label_encoder.inverse_transform(predictions)
for label in predicted_labels:
    print(label)

# Make predictions for DNS queries
X_new = ["example.com", "google.com", "facebook.com"]  # Replace with your DNS queries
X_new_tokens = tokenizer.batch_encode_plus(
    X_new,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

X_new_input_ids = X_new_tokens['input_ids'].to(device)
X_new_attention_mask = X_new_tokens['attention_mask'].to(device)

new_dataset = TensorDataset(X_new_input_ids, X_new_attention_mask)
new_dataloader = DataLoader(new_dataset, batch_size=batch_size)

predictions = predict(model, new_dataloader, device)

# Perform further analysis or actions based on the predicted labels
# For example, you can print the predicted labels
predicted_labels = label_encoder.inverse_transform(predictions)
for query, label in zip(X_new, predicted_labels):
    print(f"Query: {query} - Predicted Label: {label}")
