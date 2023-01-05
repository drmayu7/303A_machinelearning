import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset/processed/breast-cancer-wisconsin-imputed.csv")

# Remove the sample_id column
df = df.drop(columns=['sample_id'])

# Split the dataset into features/variables (X) and labels/outcome (y)
X = df.drop(columns=['class'])
y = df['class']

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Run supervised learning algorithm
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Visualize the confusion matrix using a heatmap
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, X_test, y_test)

# Calculate the accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Calculate the f1-score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, pos_label=2)
print("F1_score:", f1)

# Combine the true labels and predicted labels into a dataframe
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
