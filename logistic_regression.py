import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns

# Read imputed datasets (MICE)
df = pd.read_csv("Dataset/processed/breast-cancer-wisconsin-imputed.csv")

# Remove the sample_id column
df = df.drop(columns=['sample_id'])

# Split the dataset into features/variables (X) and labels/outcome (y)
X = df.drop(columns=['class'])
y = df['class']

# Scale the features - Min and Max scaling (sensitive to outliers)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Cross-Validation for small datasets to prevent biased and overfitting of model
loo = LeaveOneOut()

# Run supervised learning - Logistic Regression algorithm
model = LogisticRegression()

# LOO method use multiple observations of evaluation, hence need to initialize a list to store 'class' prediction from all observations
predictions = []

# Loop through the cross-validation splits/observations
for train_index, test_index in loo.split(X_scaled):
    # While splitting data into training and test sets
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data and predict the labels for the test set
    model.fit(X_train,y_train)
    predicted_y = model.predict(X_test)

    # Append each predicted label and accuracy score into list
    predictions.append(predicted_y[0])

# Convert the list to a Numpy Array
predictions = np.array(predictions)

# Calculate confusion matrix
conf_mtx = confusion_matrix(y,predictions)

# Plot confusion matrix using Seaborn Heatmap
sns.heatmap(conf_mtx,annot=True,fmt='d',cmap='viridis')

# Print the classification report
print(classification_report(y, predictions))

# Combine the true labels and predicted labels into a dataframe
df2 = pd.DataFrame({'Actual': y, 'Predicted': predictions})
