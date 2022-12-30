#!/usr/bin/env python3

# NEURAL NETWORK IMPLEMENTATION - TESTS
# 2022 (c) Micha Johannes Birklbauer
# https://github.com/michabirklbauer/
# micha.birklbauer@gmail.com

def test_bcc():

    #### Binary-class Classification ####

    from zipfile import ZipFile as zip

    with zip("data.zip") as f:
        f.extractall()
        f.close()

    from neuralnet import NeuralNetwork
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("binaryclass_train.csv", header = None)
    data["label"] = data[1].apply(lambda x: 1 if x == "M" else 0)
    train, test = train_test_split(data, test_size = 0.3)
    train_data = train.loc[:, ~train.columns.isin([0, 1, "label"])].to_numpy()
    train_target = train["label"].to_numpy()
    test_data = test.loc[:, ~test.columns.isin([0, 1, "label"])].to_numpy()
    test_target = test["label"].to_numpy()

    NN = NeuralNetwork(input_size = train_data.shape[1])
    NN.add_layer(16, "relu")
    NN.add_layer(16, "relu")
    NN.add_layer(1, "sigmoid")
    NN.compile(loss = "binary crossentropy")
    NN.summary()

    hist = NN.fit(train_data, train_target, epochs = 1000, batch_size = 32, learning_rate = 0.01)

    train_predictions = np.round(NN.predict(train_data))
    train_acc = accuracy_score(train["label"].to_numpy(), train_predictions)
    test_predictions = np.round(NN.predict(test_data))
    test_acc = accuracy_score(test["label"].to_numpy(), test_predictions)

    import os
    os.remove("binaryclass_train.csv")
    os.remove("multiclass_train.csv")

    assert train_acc > 0.85 and test_acc > 0.85

def test_mcc():

    #### Multi-class Classification ####

    from zipfile import ZipFile as zip

    with zip("data.zip") as f:
        f.extractall()
        f.close()

    from neuralnet import NeuralNetwork
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("multiclass_train.csv")
    train, test = train_test_split(data, test_size = 0.3)
    train_data = train.loc[:, train.columns != "label"].to_numpy() / 255
    train_target = train["label"].to_numpy()
    test_data = test.loc[:, test.columns != "label"].to_numpy() / 255
    test_target = test["label"].to_numpy()

    one_hot = OneHotEncoder(sparse = False, categories = "auto")
    train_target = one_hot.fit_transform(train_target.reshape(-1, 1))
    test_target = one_hot.transform(test_target.reshape(-1, 1))

    NN = NeuralNetwork(input_size = train_data.shape[1])
    NN.add_layer(32, "relu")
    NN.add_layer(16, "relu")
    NN.add_layer(10, "softmax")
    NN.compile(loss = "categorical crossentropy")
    NN.summary()

    hist = NN.fit(train_data, train_target, epochs = 30, batch_size = 16, learning_rate = 0.05)

    train_predictions = np.argmax(NN.predict(train_data), axis = 1)
    train_acc = accuracy_score(train["label"].to_numpy(), train_predictions)
    test_predictions = np.argmax(NN.predict(test_data), axis = 1)
    test_acc = accuracy_score(test["label"].to_numpy(), test_predictions)

    import os
    os.remove("binaryclass_train.csv")
    os.remove("multiclass_train.csv")

    assert train_acc > 0.85 and test_acc > 0.85
