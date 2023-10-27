
---

# Fake News Detection using Machine Learning

This project utilizes a machine learning model to detect fake news. It employs a Passive Aggressive classifier with a TfidfVectorizer to process text data and make predictions on news authenticity. 

## Requirements

Ensure you have the following libraries installed:

- pandas
- scikit-learn

You can install the required packages using pip:

```
pip3 install pandas scikit-learn
```

## Usage

1. Clone the repository and navigate to the project directory.

2. Run the `detect-fake-news.py` script to load the dataset, train the model, and make predictions.

```bash
python3 detect-fake-news.py
```

3. Ensure that you replace `'your-csv-path'` with the actual path to your CSV dataset.

4. The script will display the accuracy of the model and print the classification (REAL or FAKE) for each news item in the dataset.

## Dataset

The code uses the dataset from the file `fake_or_real_news.csv` 

