from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize k-NN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = knn_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Display correct and wrong predictions
correct_predictions = 0
wrong_predictions = 0

for i in range(len(y_test)):
    if predictions[i] == y_test[i]:
        print(f"Correct Prediction - Actual: {y_test[i]}, Predicted: {predictions[i]}")
        correct_predictions += 1
    else:
        print(f"Wrong Prediction - Actual: {y_test[i]}, Predicted: {predictions[i]}")
        wrong_predictions += 1

print("\nCorrect Predictions:", correct_predictions)
print("Wrong Predictions:", wrong_predictions)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)
