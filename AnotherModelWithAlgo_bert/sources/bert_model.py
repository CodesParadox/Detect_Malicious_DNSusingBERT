import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Check if GPU is available
device = torch.device("cuda")

# Load the DNS traffic dataset and preprocess the data into a dataframe
df = pd.read_csv('dns_totalPack.csv')

# Extract relevant features from the dataset
# Modify the code below to extract the appropriate features based on your dataset
X = df[{'request_ts', 'response_ts', 'client_token_dec', 'qname', 'qtype', 'response_flags', 'answer', 'nx_domains',
        'l1_cache_hit', 'response_size', 'category_id'}].values
y = df['in_BL'].values

# Split the dataset into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42).to(device)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42).to(device)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Load the BERT tokenizer
# We initialize the BERT tokenizer, specifically the 'bert-base-uncased' variant,
# which is a pre-trained BERT model with lowercase text.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input sequences
X_train_tokens = tokenizer.batch_encode_plus(
    X_train.tolist(),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
X_val_tokens = tokenizer.batch_encode_plus(
    X_val.tolist(),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
X_test_tokens = tokenizer.batch_encode_plus(
    X_test.tolist(),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Convert tokenized inputs to tensors
X_train_input_ids = X_train_tokens['input_ids'].to(device)
X_train_attention_mask = X_train_tokens['attention_mask'].to(device)
X_val_input_ids = X_val_tokens['input_ids'].to(device)
X_val_attention_mask = X_val_tokens['attention_mask'].to(device)
X_test_input_ids = X_test_tokens['input_ids'].to(device)
X_test_attention_mask = X_test_tokens['attention_mask'].to(device)

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train_encoded).to(device)
y_val_tensor = torch.tensor(y_val_encoded).to(device)
y_test_tensor = torch.tensor(y_test_encoded).to(device)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_input_ids, X_train_attention_mask, y_train_tensor)
val_dataset = TensorDataset(X_val_input_ids, X_val_attention_mask, y_val_tensor)
test_dataset = TensorDataset(X_test_input_ids, X_test_attention_mask, y_test_tensor)

# Create PyTorch data loaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)

# Set the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Set the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_loss = float('inf')
best_val_f1 = 0.0
best_epoch = 0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
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

    # Validation
    model.eval()
    total_val_loss = 0
    y_val_true = []
    y_val_pred = []

    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in val_dataloader:
            # Move inputs and labels to the GPU
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Calculate predictions
            _, predictions = torch.max(logits, dim=1)

            y_val_true.extend(batch_labels.tolist())
            y_val_pred.extend(predictions.tolist())

            total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        val_f1 = f1_score(y_val_true, y_val_pred)

        print(f'Epoch {epoch + 1} - Validation Loss: {average_val_loss:.4f} - Validation F1 Score: {val_f1:.4f}')

        # Check if this is the best validation loss or F1 score
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_val_f1 = val_f1
            best_epoch = epoch

    # Early stopping if validation loss does not improve for 3 consecutive epochs
    if epoch - best_epoch >= 3:
        print('Early stopping...')
        break

print(f'Best Validation Loss: {best_val_loss:.4f}')
print(f'Best Validation F1 Score: {best_val_f1:.4f}')

# Test the model
model.eval()
total_test_loss = 0
y_test_true = []
y_test_pred = []

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

        # Calculate predictions
        _, predictions = torch.max(logits, dim=1)

        y_test_true.extend(batch_labels.tolist())
        y_test_pred.extend(predictions.tolist())

        total_test_loss += loss.item()

average_test_loss = total_test_loss / len(test_dataloader)
test_f1 = f1_score(y_test_true, y_test_pred)

print(f'Test Loss: {average_test_loss:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')


# Calculate precision, recall, and accuracy
test_precision = precision_score(y_test_true, y_test_pred)
test_recall = recall_score(y_test_true, y_test_pred)
test_accuracy = accuracy_score(y_test_true, y_test_pred)

print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')