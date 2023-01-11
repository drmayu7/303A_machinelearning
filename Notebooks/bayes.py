import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, plot_confusion_matrix

# Load the dataset
df = pd.read_csv("Dataset/processed/breast-cancer-wisconsin-imputed.csv")

# Remove the sample_id column
df = df.drop("sample_id", axis=1)

# Split the dataset into features and target
X = df.drop("class", axis=1)
y = df["class"]

# Scale the features in the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()

# Initialize the Leave One Out splitter
loo = LeaveOneOut()

# Initialize the list to store the accuracies
predictions = []
probabilities = []

# Loop through the splits
for train_index, test_index in loo.split(X_scaled):
    # Split the data into train and test sets
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the train set
    model.fit(X_train, y_train)

    # Get prediction
    prediction = model.predict(X_test)

    # Get probability
    probability = model.predict_proba(X_test)

    # Write to append predictions and probabilities array
    predictions.append(prediction[0])
    probabilities.append(probability[0])

# Plot Confusion Matrix
plot_confusion_matrix(model, X_scaled, y)

# Calculate the accuracy of the model and display classification results
accuracy = accuracy_score(y, predictions)
print(f'The leave-one-out accuracy for breast-cancer-wisconsin dataset: {accuracy:.4f}')
print(classification_report(y,predictions))




