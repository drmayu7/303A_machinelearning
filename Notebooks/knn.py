import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt

# (suppress unnecessary warning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load the imputed dataset
df = pd.read_csv("Dataset/processed/breast-cancer-wisconsin-imputed.csv")

# Drop the 'sample_id' column
df = df.drop('sample_id', axis=1)

# Split the data into features and outcome
X = df.drop('class', axis=1)
y = df['class']

# Scale the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# classification testing
predictions = np.zeros(X.shape[0])
probabilities = np.zeros([X.shape[0], 2])
loo = LeaveOneOut()

# loop from 1 to 40 with interval of 2 (off numbers only) using LOO
k_range = range(1, 40, 2)
accuracies = np.zeros(len(k_range))
count = 0

for k in k_range:
    for train_index, test_index in loo.split(X):
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]

        # Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        # Train the model using the training sets
        knn.fit(X_train, y_train)

        # get prediction
        prediction = knn.predict(X_test)

        # write to append predictions array
        predictions[test_index] = prediction

    # report classification results
    agreement = (predictions == y).sum()
    accuracy = agreement / y.shape[0]
    print("k={0},The leave-one-out accuracy is: {1:.4f}".format(k, accuracy))
    accuracies[count] = accuracy
    count = count + 1

# plot the accuracies
plt.plot(k_range, accuracies)
plt.xlabel("Value of k")
plt.ylabel("Accuracy")




