import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras import Input
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import to_categorical


class ProcessData:
    def __init__(self, dataset):
        self.dataset = dataset
        match self.dataset:
            case "eng_alphabets":
                self.model_name = "eng_alphabets"
                self.word_dict = {
                    0: "A",
                    1: "B",
                    2: "C",
                    3: "D",
                    4: "E",
                    5: "F",
                    6: "G",
                    7: "H",
                    8: "I",
                    9: "J",
                    10: "K",
                    11: "L",
                    12: "M",
                    13: "N",
                    14: "O",
                    15: "P",
                    16: "Q",
                    17: "R",
                    18: "S",
                    19: "T",
                    20: "U",
                    21: "V",
                    22: "W",
                    23: "X",
                    24: "Y",
                    25: "Z",
                }
                self.res = 28
                self.classes = 26

            case "roman_digits":
                self.model_name = "roman_digits"
                self.word_dict = {
                    0: "0",
                    1: "1",
                    2: "2",
                    3: "3",
                    4: "4",
                    5: "5",
                    6: "6",
                    7: "7",
                    8: "8",
                    9: "9",
                }
                self.res = 28
                self.classes = 10

            case "hindi_alphabets":
                self.model_name = "hindi_alphabets"
                self.word_dict = {
                    0: "CHECK",
                    1: "ka",
                    2: "kha",
                    3: "ga",
                    4: "gha",
                    5: "kna",
                    6: "cha",
                    7: "chha",
                    8: "ja",
                    9: "jha",
                    10: "yna",
                    11: "taa",
                    12: "thaa",
                    13: "daa",
                    14: "dhaa",
                    15: "adna",
                    16: "ta",
                    17: "tha",
                    18: "da",
                    19: "dha",
                    20: "na",
                    21: "pa",
                    22: "pha",
                    23: "ba",
                    24: "bha",
                    25: "ma",
                    26: "yaw",
                    27: "ra",
                    28: "la",
                    29: "waw",
                    30: "sha",
                    31: "sha",
                    32: "sa",
                    33: "ha",
                    34: "kshya",
                    35: "tra",
                    36: "gya",
                    37: "CHECK",
                }
                self.res = 32
                self.classes = 38

            case "devnagri_digits":
                self.model_name = "devnagri_digits"
                self.word_dict = {
                    0: "0",
                    1: "1",
                    2: "2",
                    3: "3",
                    4: "4",
                    5: "5",
                    6: "6",
                    7: "7",
                    8: "8",
                    9: "9",
                }
                self.res = 32
                self.classes = 10

            case _:
                self.model_name = None
                self.word_dict = None
                self.res = None
                self.classes = None

    def process(self, data):
        dataframe = pd.read_csv(data).astype("float32")
        X = dataframe.drop("0", axis=1)
        y = dataframe["0"]

        # Reshaping the data in csv file so that it can be displayed as an image...
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        train_x = np.reshape(train_x.values, (train_x.shape[0], self.res, self.res))
        test_x = np.reshape(test_x.values, (test_x.shape[0], self.res, self.res))

        print("Train data shape: ", train_x.shape)
        print("Test data shape: ", test_x.shape)
        print(train_y.shape)
        print(test_y.shape)

        train_yint = np.int0(y)
        count = np.zeros(self.classes, dtype="int")
        for i in train_yint:
            count[i] += 1

        alphabets = []
        for i in self.word_dict.values():
            alphabets.append(i)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.barh(alphabets, count)

        plt.xlabel("Number of elements ")
        plt.ylabel("Alphabets")
        plt.grid()
        plt.show()

        # Shuffling the data ...
        shuff = shuffle(train_x[:100])

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        axes = ax.flatten()

        for i in range(9):
            axes[i].imshow(np.reshape(shuff[i], (self.res, self.res)), cmap="Greys")
        plt.show()

        # Reshaping the training & test dataset so that it can be put in the model...
        train_X = train_x.reshape(
            train_x.shape[0], train_x.shape[1], train_x.shape[2], 1
        )
        print("New shape of train data: ", train_X.shape)

        test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
        print("New shape of test data: ", test_X.shape)

        # Converting the labels to categorical values...
        train_yOHE = to_categorical(train_y, num_classes=self.classes, dtype="int")
        print("New shape of train labels: ", train_yOHE.shape)

        test_yOHE = to_categorical(test_y, num_classes=self.classes, dtype="int")
        print("New shape of test labels: ", test_yOHE.shape)

        return (
            train_X,
            train_yOHE,
            test_X,
            test_yOHE,
            self.res,
            self.classes,
            self.model_name,
        )


class TrainModel:
    def __init__(self, dataset, trainable=True):
        self.TRAINING = trainable
        self.dataset = dataset
        currentdir = os.path.dirname(os.getcwd())
        self.modelpath = os.path.join(
            currentdir, "models", "model_" + self.dataset + ".h5"
        )
        self.logdir = os.path.join(currentdir, "logs" + self.dataset)
        self.data = os.path.join(currentdir, "data", self.dataset + ".csv")

    def train(self):
        process_data = ProcessData(dataset=self.dataset)
        (
            train_X,
            train_yOHE,
            test_X,
            test_yOHE,
            res,
            classes,
            model_name,
        ) = process_data.process(data=self.data)

        if self.TRAINING == "True":
            input = Input(shape=(None, res, res, 1))
            x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(input)
            x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
            x = Conv2D(
                filters=64, kernel_size=(3, 3), activation="relu", padding="same"
            )(x)
            x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
            x = Conv2D(
                filters=128, kernel_size=(3, 3), activation="relu", padding="valid"
            )(x)
            x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
            x = Flatten()(x)
            x = Dense(64, activation="relu")(x)
            x = Dense(128, activation="relu")(x)
            output = Dense(classes, activation="softmax", dtype="float32")(x)

            model = keras.Model(inputs=input, outputs=output)

            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            model.summary()

            callbacks = [
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001
                ),
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=2,
                    verbose=0,
                    mode="auto",
                ),
                TensorBoard(log_dir=self.logdir),
            ]

            history = model.fit(
                train_X,
                train_yOHE,
                epochs=10,
                callbacks=callbacks,
                validation_split=0.2,
                use_multiprocessing=True,
            )

            model.save(self.modelpath)

            # Displaying the accuracies & losses for train & validation set...
            print("The validation accuracy is :", history.history["val_accuracy"])
            print("The training accuracy is :", history.history["accuracy"])
            print("The validation loss is :", history.history["val_loss"])
            print("The training loss is :", history.history["loss"])

        else:
            model = tf.keras.models.load_model(self.modelpath)
            model.summary()


if __name__ == "__main__":
    training = sys.argv[1]  # Change this to False if you want to use trained model after initial training
    dataset = sys.argv[2]  # Set the dataset name

    train_model = TrainModel(dataset=dataset, trainable=training)
    train_model.train()
