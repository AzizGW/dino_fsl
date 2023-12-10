import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,precision_score,recall_score
import torch

def testTheModel(test_support_loader, test_query_loader, best_model,device):
    """
    Evaluate the model on the test set using support and query loaders.

    Args:
        test_support_loader: DataLoader for the support set of the test data.
        test_query_loader: DataLoader for the query set of the test data.
        best_model: The trained MatchingNetworks model with the best state.
    """

    best_model.eval()  # Set the model to evaluation mode

    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for support_data, query_data in zip(test_support_loader, test_query_loader):
            support_images, support_labels = support_data
            query_images, query_labels = query_data

            # Move data to the configured device
            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)

            # Process the support set
            best_model.process_support_set(support_images, support_labels)

            # Predict query labels
            log_probabilities = best_model(query_images)

            # Convert log probabilities to actual class predictions
            _, predicted = torch.max(log_probabilities.exp(), 1)

            # Store predictions and true labels
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(query_labels.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_true_labels = np.array(test_true_labels)

    accuracy = accuracy_score(test_true_labels, test_predictions)
    cm = confusion_matrix(test_true_labels, test_predictions)

    # Display test results
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Precision: {precision_score(test_true_labels, test_predictions, average='macro')*100:.2f}%")
    print(f"Test Recall: {recall_score(test_true_labels, test_predictions, average='macro')*100:.2f}%")
    print(f"Test F1: {f1_score(test_true_labels, test_predictions, average='macro')*100:.2f}%")

    print("Test confusion matrix:")
    print(cm)
    
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()"""
    

    return accuracy
