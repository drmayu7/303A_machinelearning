import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Read imputed datasets (MICE)
df = pd.read_csv("Dataset/processed/breast-cancer-wisconsin-imputed.csv")

# Remove the sample_id column
df = df.drop(columns=['sample_id'])

# Split the dataset into features/variables (X) and labels/outcome (y)
# Change values of 'class' from [2,4] to [0,1]
X = df.drop(columns=['class'])
y = df['class'].apply(lambda x: 1 if x == 4 else 0)

# Scale the features - Min and Max scaling (sensitive to outliers)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Cross-Validation for small datasets to prevent biased and overfitting of model
loo = LeaveOneOut()

# Run supervised learning - Logistic Regression algorithm
model = LogisticRegression()

# LOO method use multiple observations of evaluation, hence need to initialize a list to store 'class' prediction and probability from all observations
predictions = []
probabilities = []

# Loop through the cross-validation splits/observations
for train_index, test_index in loo.split(X_scaled):
    # While splitting data into training and test sets
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data and predict the labels + probability for the test set
    model.fit(X_train,y_train)
    predicted_y = model.predict(X_test)
    probability_y = model.predict_proba(X_test)[:,1]

    # Append each predicted label and probability score into list
    predictions.append(predicted_y[0])
    probabilities.append(probability_y[0])

# Convert the lists to a Numpy Array
predictions = np.array(predictions)
probabilities = np.array(probabilities)

# Calculate false +ve rate, true +ve rate and threshold values
fpr, tpr, thresholds = roc_curve(y,probabilities)

# Calculate area under the curve
roc_auc = roc_auc_score(y,probabilities)

# Plot the ROC curve
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Calculate confusion matrix
conf_mtx = confusion_matrix(y,predictions)

# Plot confusion matrix using Seaborn Heatmap
sns.heatmap(conf_mtx,annot=True,fmt='d',cmap='viridis')

# Print the classification report
print(classification_report(y, predictions))

# Combine the true labels and predicted labels into a dataframe
df2 = pd.DataFrame({'Actual': y, 'Predicted': predictions})
