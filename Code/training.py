import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import gc

def train(matching_networks, support_loader, query_loader, num_epochs,device):
    """
    Train the MatchingNetworks model and save the best model state.

    Args:
        matching_networks: The MatchingNetworks model with DinoViTEmbedding as the backbone.
        support_loader: DataLoader for the support set.
        query_loader: DataLoader for the query set.
        num_epochs: Number of epochs to train the model.
        device: The device (CPU or GPU) to use for training.

    Returns:
        best_model_state: The state of the best performing model.
    """
    #, weight_decay=1e-5

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(matching_networks.parameters(), lr=0.001)
    
    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_training_loss = 100000.00
    best_model_state = None

    matching_networks.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for support_data, query_data in zip(support_loader, query_loader):
            support_images, support_labels = support_data
            query_images, query_labels = query_data

            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)

            optimizer.zero_grad()

            # Process the support set
            matching_networks.process_support_set(support_images, support_labels)

            # Predict query labels and compute loss
            query_log_probabilities = matching_networks(query_images)
            loss = criterion(query_log_probabilities, query_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_labels = torch.max(query_log_probabilities.exp(), 1)
            correct_predictions += (predicted_labels == query_labels).sum().item()
            total_predictions += query_labels.size(0)
            # clear cache
            gc.collect()

        training_loss = running_loss / len(query_loader)
        scheduler.step(training_loss)

        # Check performance and save best model
        epoch_accuracy = (correct_predictions / total_predictions) * 100
        if training_loss < best_training_loss:
            best_training_loss = training_loss
            best_model_state = matching_networks.state_dict()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {training_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

    return best_model_state,best_training_loss
