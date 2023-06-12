
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

def classification_model_bert(df_all):
    model_name = 'bert-base-uncased'  # Example model name
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    max_length = 15


    # Assuming you have your dataset as a list of texts and corresponding labels
    # texts = list_answer  # List of texts
    texts = df_all["formatted_ans"].to_list()  # List of texts

    labels = [int(label) for label in df_all["is_malicious"].to_list()]  # List of labels
    texts, labels = shuffle_lists(texts, labels)

    # Tokenize the texts and convert them to input tensors
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)



    # Define the split ratio
    train_ratio = 0.8
    train_size = int(train_ratio * len(texts))

    # Split the dataset
    train_inputs, val_inputs = input_ids[:train_size], input_ids[train_size:]
    train_masks, val_masks = attention_masks[:train_size], attention_masks[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    # Create PyTorch DataLoader for training and validation sets
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    batch_size = 32 #changed from 16 -> 8 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Move the model to the device
    model.to(device)

    # Define the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}')

    # Save the fine-tuned model
    model.save_pretrained('/media/hodefi/q/DNS_DL/code/model/bert_model2.sav')

    
def train_random_forest_model(X, y, model_save_file_name):
    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    filename = model_save_file_name
    with open(filename, 'wb') as file:
        pickle.dump(rf_model, file)


def train_isolation_model_article(df, model_save_file_name="model/model_isolation_article.sav"):
    # select the columns you want to use as features, this feature name are the name of the columns in df_include_added_feature.pkl file
    X = df[['entropy', 'type_ratio', 'qname_vol', 'qname_len', 'unique_query_ratio', 'avg_longest_word_len']]

    # select the column you want to predict
    y = df['is_malicious']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the isolation forest model
    clf = IsolationForest(random_state=42)
    clf.fit(X_train)

    filename = model_save_file_name
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)


def train_random_forest_model_article(df, model_save_file_name="model/model_random_forest_article.sav"):
    # Define your feature columns and target column
    X = df[['entropy', 'type_ratio', 'qname_vol', 'qname_len', 'unique_query_ratio', 'avg_longest_word_len']]

    y = df['is_malicious']
    train_random_forest_model(X, y, model_save_file_name)


def train_ann(df, new_df_embading, model_save_file_name="model/model_ann_bert.sav"):
    X = new_df_embading
    y = df['is_malicious']

    # Split your data into a training set and a validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Change num_classes to 1 and activation to 'sigmoid'

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])  # Change loss to 'binary_crossentropy'

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

    # Save the model to a file
    with open(model_save_file_name, 'wb') as file:
        pickle.dump(model, file)


def train_random_forest_model_euclidean_bert(df,
                                             model_save_file_name="model/model_random_forest_euclidean_bert_distance.sav"):
    # Define your feature columns and target column
    X = df[['euclidean_distance_emmbading']]

    y = df['is_malicious']

    train_random_forest_model(X, y, model_save_file_name)


def train_random_forest_model_manhattan_bert(df,
                                             model_save_file_name="model/model_random_forest_manhattan_bert_distance_n.sav"):
    # Define your feature columns and target column
    X = df[['manhattan_distance_emmbading']]

    y = df['is_malicious']

    train_random_forest_model(X, y, model_save_file_name)
