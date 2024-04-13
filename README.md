# Social Media Sentiment Analysis

## Overview

This project aims to perform sentiment analysis on Twitter data using Natural Language Processing (NLP) techniques and machine learning algorithms. The dataset used for this project is the Sentiment140 dataset, which contains 1.6 million tweets labeled with sentiment.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib (optional for visualization)

## Setup

1. **Install required packages**:
    ```bash
    ! pip install kaggle
    ```

2. **Configure Kaggle API**:
    ```bash
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```

3. **Download the Sentiment140 dataset**:
    ```bash
    !kaggle datasets download -d kazanova/sentiment140
    ```

4. **Extract the dataset**:
    ```python
    from zipfile import ZipFile
    dataset = '/content/sentiment140.zip'

    with ZipFile(dataset, 'r') as zip:
        zip.extractall()
        print('The dataset is extracted')
    ```

## Data Preprocessing

- Load the dataset using Pandas.
- Replace the target values (`4` with `1`) to convert the sentiment labels to binary (`0` for negative, `1` for positive).
- Perform text preprocessing, including stemming and removing stopwords.

## Feature Extraction

- Use TF-IDF Vectorizer to convert text data into numerical vectors.

## Model Training

- Train a Logistic Regression model on the TF-IDF vectors.
- Evaluate the model's accuracy on both training and test datasets.

## Results

- The trained model achieved an accuracy score of approximately `81.0%` on the training data and `77.8%` on the test data.

## Prediction

- Save the trained model using pickle.
- Load the saved model to make predictions on new data.

## Usage

To run the project:

1. Open the provided Jupyter notebook (`Senti.ipynb`) in Google Colab or any Jupyter notebook environment.
2. Follow the code cells step by step to execute the data preprocessing, feature extraction, model training, and prediction steps.
