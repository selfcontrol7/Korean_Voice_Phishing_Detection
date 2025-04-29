# multimodal_fusion_model.py

import torch
import torch.nn as nn # Importing the PyTorch library for neural networks


class MultimodalFusionModel(nn.Module):
    def __init__(self, audio_dim=869, text_dim=768, use_both_text=False, hidden_dim=512, dropout=0.3, *args, **kwargs):
        """
        Initializes the MultimodalFusionModel class. This model is designed to combine audio and text features
        for a fusion-based classification task. It provides configurable parameters for feature dimensions,
        layer sizes, dropout rate, and the ability to use both text features.

        Attributes
        ----------
        use_both_text : bool
            Boolean flag indicating whether to use both text features for fusion.
        audio_dim : int
            Dimension of the input audio feature vector.
        text_dim : int
            Dimension of the input text feature vector.
        total_text_dim : int
            Total dimension of text feature vectors. Doubled if both text features are used.
        audio_projection : torch.nn.Linear
            Linear projection layer for mapping audio features to the hidden dimension.
        text_projection : torch.nn.Linear
            Linear projection layer for mapping text features to the hidden dimension.
        droupout : torch.nn.Dropout
            Dropout layer applied to reduce overfitting during training.
        fusion_layer : torch.nn.Linear
            Linear layer for fusing the audio and text representations.
        classifier : torch.nn.Linear
            Linear layer for producing output classification logits.
        relu : torch.nn.ReLU
            Activation function applied after certain layers.

        Parameters
        ----------
        audio_dim : int, optional
            Dimension of the input audio feature vector (default is 869).
        text_dim : int, optional
            Dimension of the input text feature vector (default is 768).
        use_both_text : bool, optional
            Specifies whether to use both text features for fusion (default is False).
        hidden_dim : int, optional
            Dimension of the hidden layer (default is 512).
        dropout : float, optional
            Dropout rate for dropout layer (default is 0.3).
        *args, **kwargs : tuple, dict
            Additional arguments for extending parent class initialization.
        """

        # Call the parent class constructor
        super().__init__(*args, **kwargs) # Initialize the parent class using the provided arguments (modern Python 3+ style)
        # super(MultimodalFusionModel, self).__init__() # Initialize the parent class using the provided arguments (legacy Python 2 style)

        # Initialize the model parameters
        self.use_both_text = use_both_text # Store whether to use both text features
        self.audio_dim = audio_dim # Store the audio dimension
        self.text_dim = text_dim # Store the textdimension
        self.total_text_dim = text_dim * 2 if use_both_text else text_dim # Total text dimension is doubled if both text features are used

        # Initialize the projection layers
        self.audio_projection = nn.Linear(audio_dim, hidden_dim) # Linear layer to project audio features to hidden dimension
        self.text_projection = nn.Linear(self.total_text_dim, hidden_dim) # Linear layer to project text features to hidden dimension

        self.droupout = nn.Dropout(dropout) # Dropout layer to prevent overfitting
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim) # Linear layer to fuse audio and text features
        self.classifier = nn.Linear(hidden_dim, 1) # Linear layer for classification

        # Initialize the activation function
        self.relu = nn.ReLU()

    def forward(self, audio_features, text_features):
        """
        Performs forward pass for a fusion model combining audio and text features. Before classification, the method processes
        audio and text features through separate projection layers, concatenates them, applies
        dropouts, projects them into a common hidden space, and applies ReLU activation
        to introduce non-linearity and help with feature transformation. ReLU also helps prevent the
        vanishing gradient problem by allowing positive gradients to flow through the network while
        setting negative values to zero, which creates sparse activations. The forward
        method outputs the final logits.

        Arguments:
            audio_features: Input feature tensor from the audio processor.
            text_features: Input feature tensor from the text processor.

        Returns:
            Tensor: Final output logits after processing.

        # Project audio and text features to a common hidden space and apply ReLU to:
        # 1. Add non-linearity for better feature transformation
        # 2. Prevent vanishing gradients by allowing positive gradients to flow
        # 3. Create sparse activations by setting negative values to zero
        """
        audio_proj = self.relu(self.audio_projection(audio_features))
        text_proj = self.relu(self.text_projection(text_features))

        # Concatenate the projected audio and text features
        fused = torch.cat([audio_proj, text_proj], dim=1)

        # Apply dropout to the concatenated features
        fused = self.droupout(fused)
        # Project the concatenated features to a common hidden space
        fused = self.relu(self.fusion_layer(fused))
        # Apply dropout to the fused features
        fused = self.droupout(fused)
        # Apply the classifier to get the final logits
        output = self.classifier(fused) # Get the final output logits

        return output.squeeze(1) # Squeeze the output to remove the extra dimension

# Example of how to use the model
if __name__ == "__main__":
    model = MultimodalFusionModel(audio_dim=869, text_dim=768, use_both_text=False) # Create an instance of the model

    batch_size = 16 # Define the batch size

    # Example input tensors
    dummy_audio_features = torch.randn(batch_size, 869) # Create random input tensors for testing
    dummy_text_features = torch.randn(batch_size, 768) # Create random input tensors for testing

    # Forward pass through the model
    output_logits = model(dummy_audio_features, dummy_text_features) # Get the output from the model (Logits is the unnormalized final scores of your model.)
    print(output_logits.shape) # Print the shape of the output logits`
    # The output shape should be (batch_size, 1) since we are using a binary classification task
