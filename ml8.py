from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample dataset
data = {
    'text': ['This is a sample document.',
             'Another document for testing.',
             'Sample text for classification.',
             'Not relevant.',
             'A document for the task.'],
    'label': ['A', 'B', 'A', 'B', 'A']
}

df = pd.DataFrame(data)

# Vectorizing text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Creating and training the Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predicting labels for the test set
y_pred = nb_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculating precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculating recall
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
