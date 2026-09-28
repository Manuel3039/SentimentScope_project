# SentimentScope

SentimentScope is a transformer-based sentiment analysis project for the IMDB movie review dataset. The project loads review text, preprocesses it, trains a compact neural model, evaluates its hold-out performance, and reports the final classification accuracy.

## Overview

This repository contains an executed notebook, `SentimentScope_executed.ipynb`, which demonstrates a complete machine learning workflow for binary sentiment classification:

- Data loading from the IMDB archive
- Exploratory data analysis
- Tokenization and train/validation splitting
- Transformer-based model construction
- Training and validation loop
- Final test evaluation and reporting

## Project Goals

The model is designed to classify movie reviews as either:

- Positive
- Negative

The notebook enforces a minimum test accuracy threshold of 75%, indicating that the model should generalize effectively beyond the training data.

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Repository Contents

- `SentimentScope_executed.ipynb` — complete executed analysis and model training workflow
- `LICENSE.txt` — project licensing information
- `CODEOWNERS` — repository ownership and review policy

## Setup

1. Open the project folder in a Python environment.
2. Install the required dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

3. Open `SentimentScope_executed.ipynb` in Jupyter Notebook or VS Code with notebook support.

## Usage

Run the notebook cells in order to:

1. Load and inspect the IMDB dataset.
2. Build the tokenizer and dataloaders.
3. Define the transformer classifier.
4. Train and validate the model.
5. Evaluate the final test performance.

## Model Summary

The model uses a compact transformer architecture with:

- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Residual feed-forward blocks
- Masked mean pooling
- Binary classification head

This design allows the model to capture sentiment patterns while remaining lightweight and efficient for training.

## Results

The final notebook prints the held-out test accuracy and asserts that it exceeds 75%. This confirms that the trained model delivers useful sentiment classification performance on unseen data.

## License

This project is distributed under the terms of the license included in `LICENSE.txt`.

## Contact

For questions or contributions, please use the project repository context or discuss with the repository owner.
