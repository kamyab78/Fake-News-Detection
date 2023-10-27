import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from the CSV file
df = pd.read_csv('your-csv-path')  # Replace with the path to your dataset

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(accuracy*100, 2)}%')

for index, news in df.iterrows():
    news_tfidf = tfidf_vectorizer.transform([news['text']])
    prediction = pac.predict(news_tfidf)
    print(f"News {index + 1}: {prediction[0]}")