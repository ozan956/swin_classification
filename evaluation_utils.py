# evaluation_utils.py
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(true_labels, predicted_labels):
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(len(class_names), len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
