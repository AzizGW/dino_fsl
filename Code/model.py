import torch
import torch.nn as nn
from transformers import ViTModel
class DinoViTEmbedding(nn.Module):
    def __init__(self, pretrained_model_name, embedding_dim,dropout_rate=0.6):
        """
        Initialize the DinoViTEmbedding class.

        Args:
            pretrained_model_name (str): Name of the pre-trained DINO ViT model.
            embedding_dim (int): The size of the output embedding dimension.
        """
        super(DinoViTEmbedding, self).__init__()

        # Load the pre-trained DINO ViT model
        self.vit_model = ViTModel.from_pretrained(pretrained_model_name)

        # Add a custom linear layer to project the embeddings to the desired dimension
        self.projection = nn.Linear(self.vit_model.config.hidden_size, embedding_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Freeze the pre-trained layers
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def forward(self, x):

        # Extract the last hidden state from the base ViT model
        with torch.no_grad():
            outputs = self.vit_model(x)
            last_hidden_state = outputs.last_hidden_state

        # We use the embeddings of the [CLS] token, which is the first token
        cls_embeddings = last_hidden_state[:, 0, :]
        
        # Apply dropout
        cls_embeddings = self.dropout(cls_embeddings)


        # Project the embeddings to the desired dimension
        projected_embeddings = self.projection(cls_embeddings)

        return projected_embeddings

    def compute_features(self, x):
        """
        Compute features for input x, an alias for the forward method.
        """
        return self.forward(x)

