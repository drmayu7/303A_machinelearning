import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load the dataset and drop the 'sample_id' column
df = pd.read_csv("Dataset/processed/breast-cancer-wisconsin-imputed.csv")
df = df.drop(columns=['sample_id'])

# Split the dataset into features and outcome
X = df.drop(columns=['class'])
y = df['class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the SVM model
model = SVC(probability=True)

# Initialize the Leave One Out cross-validator
loo = LeaveOneOut()

# Initialize a list to store the predictions made by the model
predictions = []

# Loop through each train/test split of the data
for train_index, test_index in loo.split(X_scaled):
    # Split the data into train and test sets
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data and make a prediction on the test data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Store the prediction
    predictions.append(y_pred[0])

# Calculate the accuracy of the model and display classification results
accuracy = accuracy_score(y, predictions)
print(f'The leave-one-out accuracy for breast-cancer-wisconsin dataset: {accuracy:.4f}')
print(classification_report(y,predictions))