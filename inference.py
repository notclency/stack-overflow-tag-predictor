# classifier.py
"""
Stack Overflow Question Tag Classifier

This module implements a deep learning classifier that predicts the most appropriate
tag for Stack Overflow questions based on their titles. It uses a hybrid model that
combines word embeddings and character-level features through a CNN, followed by a
bidirectional LSTM with attention.

Main function:
    classify(titles) - Predicts tags for a list of question titles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
import nltk
from nltk.stem import WordNetLemmatizer
import os

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters,
    and applying lemmatization.

    Args:
        text (str): Input text string

    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9\s+#]", "", text.lower())

    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


class AttentionLayer(nn.Module):
    """
    Attention mechanism that computes a weighted sum of LSTM outputs.

    Args:
        hidden_dim (int): Dimension of the hidden state
    """

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()

        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output):
        """
        Calculate attention weights and context vector.

        Args:
            lstm_output (torch.Tensor): Output from LSTM [batch_size, seq_len, hidden_dim * 2]

        Returns:
            tuple: (context_vector, attention_weights)
        """

        energy = torch.tanh(self.attention(lstm_output))
        attention_weights = F.softmax(energy, dim=1)

        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights.squeeze(2)


class EnhancedTextModel(nn.Module):
    """
    Neural network model for text classification that combines:
    1. Word embeddings
    2. Character-level CNN
    3. Bidirectional LSTM
    4. Attention mechanism

    Args:
        word_vocab_size (int): Size of the word vocabulary
        char_vocab_size (int): Size of the character vocabulary
        word_embed_dim (int): Dimension of word embeddings
        char_embed_dim (int): Dimension of character embeddings
        char_cnn_out_channels (int): Output channels from character CNN
        hidden_dim (int): Hidden dimension for LSTM
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for regularization
        word_pad_idx (int): Padding index for words
        char_pad_idx (int): Padding index for characters
    """

    def __init__(self, word_vocab_size, char_vocab_size, word_embed_dim, char_embed_dim,
                 char_cnn_out_channels, hidden_dim, num_layers, num_classes,
                 dropout=0.5, word_pad_idx=0, char_pad_idx=0):
        super(EnhancedTextModel, self).__init__()

        self.word_pad_idx = word_pad_idx
        self.char_pad_idx = char_pad_idx

        # Word Embedding Layer
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=word_pad_idx)

        # Character Embedding Layer
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=char_pad_idx)

        self.char_cnn = nn.Conv1d(in_channels=char_embed_dim,
                                  out_channels=char_cnn_out_channels,
                                  kernel_size=3,
                                  padding=1)

        lstm_input_dim = word_embed_dim + char_cnn_out_channels

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention Layer
        self.attention = AttentionLayer(hidden_dim)

        # Fully Connected Layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        """
        Forward pass through the model.

        Args:
            inputs (tuple): (word_inputs, char_inputs)
                word_inputs: [batch_size, seq_len]
                char_inputs: [batch_size, seq_len, word_len]

        Returns:
            torch.Tensor: Output logits [batch_size, num_classes]
        """
        word_inputs, char_inputs = inputs
        batch_size, seq_len = word_inputs.shape
        word_len = char_inputs.shape[2]

        word_embedded = self.word_embedding(word_inputs)

        char_embedded = self.char_embedding(char_inputs)
        char_embedded_reshaped = char_embedded.view(batch_size * seq_len, word_len, -1)
        char_embedded_permuted = char_embedded_reshaped.permute(0, 2, 1)

        char_cnn_out = F.relu(self.char_cnn(char_embedded_permuted))
        char_features = F.max_pool1d(char_cnn_out, char_cnn_out.shape[2]).squeeze(2)
        char_features_reshaped = char_features.view(batch_size, seq_len, -1)

        combined_embedded = torch.cat((word_embedded, char_features_reshaped), dim=2)
        combined_embedded = self.dropout_layer(combined_embedded)

        lstm_output, _ = self.lstm(combined_embedded)

        context_vector, _ = self.attention(lstm_output)

        x = self.fc1(context_vector)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.layer_norm(x)
        x = self.fc2(x)

        return x


class TextDataset(Dataset):
    """
    Dataset class for text classification that processes both word-level
    and character-level features.

    Args:
        texts (list): List of text samples
        labels (list): List of corresponding labels
        word_vocab (dict): Word-to-index vocabulary mapping
        char_vocab (dict): Character-to-index vocabulary mapping
        max_seq_len (int): Maximum sequence length for padding/truncating
        max_word_len (int): Maximum word length for padding/truncating characters
    """

    def __init__(self, texts, labels, word_vocab, char_vocab, max_seq_len, max_word_len):
        self.texts = texts
        self.labels = labels
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        self.word_pad_idx = word_vocab['<pad>']
        self.word_unk_idx = word_vocab['<unk>']
        self.char_pad_idx = char_vocab['<pad>']
        self.char_unk_idx = char_vocab['<unk>']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single processed sample.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: ((word_indices, char_indices), label)
        """
        text = self.texts[idx]
        label = self.labels[idx]
        words = text.split()

        word_indices = [self.word_vocab.get(word, self.word_unk_idx) for word in words]

        char_indices_word_list = []
        for word in words:

            chars = [self.char_vocab.get(char, self.char_unk_idx) for char in word]

            if len(chars) > self.max_word_len:
                chars = chars[:self.max_word_len]
            else:
                chars += [self.char_pad_idx] * (self.max_word_len - len(chars))
            char_indices_word_list.append(chars)

        seq_len = len(word_indices)
        if seq_len > self.max_seq_len:
            word_indices = word_indices[:self.max_seq_len]
            char_indices_word_list = char_indices_word_list[:self.max_seq_len]
        else:
            word_indices += [self.word_pad_idx] * (self.max_seq_len - seq_len)

            pad_chars = [self.char_pad_idx] * self.max_word_len
            char_indices_word_list += [pad_chars] * (self.max_seq_len - seq_len)

        return (torch.tensor(word_indices, dtype=torch.long),
                torch.tensor(char_indices_word_list, dtype=torch.long)), \
            torch.tensor(label, dtype=torch.long)


def classify(titles: list[str]) -> list[str]:
    """
    Predicts the most appropriate Stack Overflow tag for each title.

    Args:
        titles (list): List of Stack Overflow question titles

    Returns:
        list: Predicted tags for each title (empty string for invalid inputs)
    """

    possible_paths = [
        "stackoverflow_classifier.pt",  # Current directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "stackoverflow_classifier.pt"),  # Parent dir
        os.path.join(os.path.dirname(__file__), "stackoverflow_classifier.pt")  # Module dir
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("Error: Model file not found in any of the expected locations")
        return [""] * len(titles)  # Return empty tags if model not found

    try:
        # Load the model
        checkpoint = torch.load(model_path, weights_only=False)

        # Extract model parameters
        word_vocab = checkpoint['word_vocab']
        char_vocab = checkpoint['char_vocab']
        label_encoder_classes = checkpoint['label_encoder_classes']
        max_seq_len = checkpoint['max_seq_len']
        max_word_len = checkpoint['max_word_len']

        # Creating the model
        model = EnhancedTextModel(
            word_vocab_size=len(word_vocab),
            char_vocab_size=len(char_vocab),
            word_embed_dim=checkpoint['word_embed_dim'],
            char_embed_dim=checkpoint['char_embed_dim'],
            char_cnn_out_channels=checkpoint['char_cnn_out_channels'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            num_classes=len(label_encoder_classes),
            dropout=checkpoint['dropout'],
            word_pad_idx=word_vocab['<pad>'],
            char_pad_idx=char_vocab['<pad>']
        ).to(device)

        # Load model weights and set to evaluation mode
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # Preprocess titles
        cleaned_titles = [clean_text(title) for title in titles]
        valid_titles = [title for title in cleaned_titles if title]  # Filter out empty titles

        if not valid_titles:
            return [""] * len(titles)  # Return empty tags for all if no valid titles

        # dataset for inference
        batch_size = 32  # Default batch size
        inference_dataset = TextDataset(
            valid_titles,
            [0] * len(valid_titles),
            word_vocab,
            char_vocab,
            max_seq_len,
            max_word_len
        )

        # dataloader for inference
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

        # Process all titles
        all_predictions = []
        with torch.no_grad():
            for inputs, _ in inference_loader:
                word_inputs, char_inputs = inputs
                word_inputs, char_inputs = word_inputs.to(device), char_inputs.to(device)

                # Forward pass
                outputs = model((word_inputs, char_inputs))
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())

        predicted_tags = [label_encoder_classes[idx] for idx in all_predictions]

        # Handle case where some titles were empty
        if len(predicted_tags) < len(titles):

            valid_predictions = {}
            valid_idx = 0
            for i, title in enumerate(titles):
                if clean_text(title):
                    valid_predictions[i] = predicted_tags[valid_idx]
                    valid_idx += 1

            final_predictions = []
            for i in range(len(titles)):
                if i in valid_predictions:
                    final_predictions.append(valid_predictions[i])
                else:
                    final_predictions.append("")

            return final_predictions

        return predicted_tags

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return [""] * len(titles)

    # For testing and demonstration purposes


if __name__ == "__main__":
    # Sample Stack Overflow question titles
    test_titles = [
        "How to sort a dictionary by value in Python?",
        "Cannot connect to MySQL database using PHP",
        "Best practices for RESTful API design",
        "Converting string to datetime in JavaScript",
        "How to handle null pointer exception in Java",
        "React component not updating when state changes",
        "Efficient way to find maximum value in an array",
        "Docker container exits immediately after starting",
        "",  # Empty title to test handling of empty input
        "Using async/await with fetch API in JavaScript"
    ]

    # Run classifier on test titles
    print("Testing classifier with sample questions...")
    predictions = classify(test_titles)

    # Display results in a formatted table
    print("\nPrediction Results:")
    print("-" * 80)
    print(f"{'Title':<60} | {'Predicted Tag':<15}")
    print("-" * 80)

    # Display results
    for title, tag in zip(test_titles, predictions):
        display_title = (title[:57] + "...") if len(title) > 57 else title
        print(f"{display_title:<60} | {tag:<15}")

    # Print unique tags found in predictions
    unique_tags = set(tag for tag in predictions if tag)
    print("\nUnique tags predicted:", ", ".join(unique_tags))

    print(f"\nTotal titles processed: {len(test_titles)}")
    print(f"Valid predictions made: {len([p for p in predictions if p])}")