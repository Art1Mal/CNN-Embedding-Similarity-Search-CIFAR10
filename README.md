# CNN-Embedding-Similarity-Search-CIFAR10
Deep CNN trained on CIFAR-10 with feature embedding extraction and pairwise image similarity via cosine distance and t-SNE visualization.
# CNN Embedding Similarity on CIFAR-10

This project implements a deep convolutional neural network (CNN) trained on the CIFAR-10 dataset. After training, the model is used to extract embedding vectors from images, and cosine similarity is computed between all test embeddings to find the top 10 most similar image pairs.

## Features

- CNN architecture with batch normalization, dropout, and L2 regularization
- Adadelta optimizer and categorical cross-entropy loss
- Evaluation using accuracy, precision, recall, and F1-score
- Embedding extraction from the penultimate dense layer
- Cosine similarity search to identify the most similar image pairs
- 2D t-SNE projection for embedding visualization

## Visual Output

- A t-SNE plot of the learned embeddings
- Side-by-side display of the 10 most similar image pairs from the test set

## Project Structure

- `cnn_embedding_visualization_cifar10.py` – main training, evaluation, embedding extraction, and visualization code
- `requirements.txt` – dependencies for running the code

## Run Instructions

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
