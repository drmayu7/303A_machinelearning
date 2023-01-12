import pandas as pd
import numpy as np

def generateData(data1):
    # make a copy of data
    data2 = data1.copy()
    # separate labels from features
    diagnosis2 = data2['class']  # labels
    features2 = data2.drop(['class'], axis=1)
    features2_headers = ["cl_thcknss", "size_cell_un", "shape_cell_un", "marg_adhesion", "size_cell_single",
                         "bare_nucl", "bl_chrmatn", "nrml_nucleo", "mitoses"]
    mean, sigma = 0, 0.1
    # creating a noise with the same dimension as the dataset
    noise = np.random.normal(mean, sigma, features2.shape)
    features2 = features2.apply(pd.to_numeric, errors='ignore')
    features2_with_noise = features2.add(pd.DataFrame(noise, columns=features2_headers), fill_value=0)
    data2 = pd.concat([features2_with_noise,
                       pd.DataFrame(diagnosis2)], axis=1)
    return data2

def neural_network(df):
    new_data = generateData(df)
    data = df.append(new_data, ignore_index=True)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numerical = ["class", "cl_thcknss", "size_cell_un", "shape_cell_un", "marg_adhesion", "size_cell_single",
                 "bare_nucl", "bl_chrmatn", "nrml_nucleo", "mitoses"]
    data[numerical] = scaler.fit_transform(data[numerical])

    diagnosis = data['class']
    features = data.drop(['class'], axis=1)
    sqrt_features = features.copy()
    for feature_name in sqrt_features.columns:
        sqrt_features[feature_name] = np.sqrt(sqrt_features[feature_name])
    features = pd.DataFrame(sqrt_features)

    # the dataset is split into two parts: 75% for model training and 25% for model testing:
    from sklearn.model_selection import train_test_split
    # Shuffle and split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        diagnosis, test_size=0.25, random_state=42)

    # After splitting dataset, the resulting data subsets must be re-indexed as follows to avoid dataset key mismatch issue:
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    score = rfc.score(X_test, y_test)

    # a reusable function can be defined for creating new model instances
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.models import Sequential
    import keras.utils
    from keras import utils as np_utils
    def createModel():
        model = Sequential()
        model.add(Dense(9, activation='relu', input_dim=9))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu', input_shape=(9,)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', input_shape=(5,)))
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    model = createModel()
    model.fit(X_train, y_train, epochs=500, batch_size=32)

    from sklearn.model_selection import StratifiedKFold
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    X = X_train
    Y = y_train
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        model = createModel()
        model.fit(X[train], Y[train], epochs=500, batch_size=10,
                  verbose=0)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("{}: {:.2f}".format(model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("{:.2f} (+/- {:.2f})".format(np.mean(cvscores), np.std(cvscores)))

    score = model.evaluate(X_test, y_test, batch_size=32)

    # Define your architecture.
    model = Sequential()
    model.add(Dense(9, activation='relu', input_dim=9))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='relu', input_shape=(9,)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', input_shape=(5,)))

    model.summary()

    # Compile NN Model
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['acc'])  # ['binary_accuracy']

    # model.fit(X_train, y_train, epochs=800, batch_size=16) # (500, 16) = 0.974286, 32 - 0.968571
    history = model.fit(X_train, y_train, epochs=1000, batch_size=16, verbose=1)

    import matplotlib.pyplot as plt
    print(type(history.history['acc']))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Test NN Model
    score = model.evaluate(X_test, y_test, batch_size=16)  # 16 - 0.974286, 32 - 0.968571
    print("score = ", score)

    # Draw ROC Curve
    from sklearn.metrics import roc_curve, auc
    from sklearn import metrics

    y_pred = predict_prob = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

