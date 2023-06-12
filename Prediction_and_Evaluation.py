import pickle
from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import sys

from transformers import BertForSequenceClassification


# predict and evaluate the BERT classification model
def evaluate_BERT_classification_model(model_file_name , the_formatted_answer_as_list, true_labels, print_to_file=False):
    predicted_labels_list = get_prediction_BERT_classification_model(model_file_name , the_formatted_answer_as_list)
    get_visual_classification_report(true_labels, predicted_labels_list, model_file_name, print_to_file=False)
    
  
# predict the differnt formmated answer using BERT classification model
# model_file_name = '/media/hodefi/q/DNS_DL/code/model/bert_model2.sav'
def get_prediction_BERT_classification_model(model_file_name , the_formatted_answer_as_list):
  
    # Load the fine-tuned model
    model_name = 'bert-base-uncased'
    model_path = model_file_name
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    
    #Preprocess the new section of the dataset
    new_texts = the_formatted_answer_as_list  # List of texts in the new section

    # Tokenize the texts and convert them to input tensors
    new_input_ids = []
    new_attention_masks = []
    max_length = 15
    
    for text in new_texts:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length= max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
    )
        new_input_ids.append(encoded_text['input_ids'])
        new_attention_masks.append(encoded_text['attention_mask'])

    new_input_ids = torch.cat(new_input_ids, dim=0)
    new_attention_masks = torch.cat(new_attention_masks, dim=0)


    #Create a new DataLoader for the new section
    from torch.utils.data import TensorDataset, DataLoader

    new_dataset = TensorDataset(new_input_ids, new_attention_masks)
    new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate the new section using the trained model:

    model.eval()
    predictions = []

    for batch in new_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)

        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())

    predicted_labels_list = predictions.tolist()
    return predicted_labels_list
    
    
    
def load_trained_model(model_save_file_name):
    filename = model_save_file_name
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    return clf


def get_visual_classification_report(y_test, y_pred, model_name, print_to_file=False):
    print("classification report for - ", model_name)
    # evaluate the model
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Evaluate the model
    # print(confusion_matrix(y_test, y_pred))
    classification__report = classification_report(y_test, y_pred)
    print(classification__report)
    cm_array = np.array(cm)

    # Calculate micro-average precision
    total_TP = sum([cm_array[i][i] for i in range(len(cm_array))])
    total_FP = sum([sum(cm_array[i]) - cm_array[i][i] for i in range(len(cm_array))])
    micro_precision = total_TP / (total_TP + total_FP)

    # Calculate micro-average recall
    total_FN = sum([sum(cm_array[i]) - cm_array[i][i] for i in range(len(cm_array))])
    micro_recall = total_TP / (total_TP + total_FN)

    # Calculate micro-average F1-score
    micro_f1_score = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))

    print("Micro-average Precision:", micro_precision)
    print("Micro-average Recall:", micro_recall)
    print("Micro-average F1-Score:", micro_f1_score)

    # create heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

    # set axis labels
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # show plot
    plt.show()

    if print_to_file:
        with PrintToFile("classification_report" + model_name + ".txt"):
            print(cm)
            print(classification__report)
            print("Micro-average Precision:", micro_precision)
            print("Micro-average Recall:", micro_recall)
            print("Micro-average F1-Score:", micro_f1_score)


# predict and evaluate the model isolation tree and random forest
def predict_and_evaluation(model_save_file_name, X_test, y_test, print_to_file=False):
    clf = load_trained_model(model_save_file_name)
    # predict on the testing set
    y_pred = clf.predict(X_test).round()
    get_visual_classification_report(y_test.astype(int), y_pred.astype(int), model_save_file_name, print_to_file)


# predict and evaluate the ann model
def ann_prediction_and_evaluation(model_save_file_name, X_test, y_test, print_to_file=False):
    clf = load_trained_model(model_save_file_name)
    y_pred = clf.predict(X_test).round()
    y_pred_classes = np.round(y_pred)  # Convert predicted probabilities to binary classes (0 or 1)
    get_visual_classification_report(y_test.astype(int), y_pred_classes.astype(int), model_save_file_name,
                                     print_to_file)


# Create a context manager to redirect prints to a file
class PrintToFile:
    def __init__(self, filename):
        self.filename = filename
        self.terminal = sys.stdout

    def __enter__(self):
        self.file = open(self.filename, 'w')
        sys.stdout = self.file

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.file.close()
