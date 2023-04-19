from threading import main_thread
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


def cnn_model(num_hidden):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv2D(
            num_hidden,
            kernel_size=4,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_hidden, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


def draw_each():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    num_hidden_list = [16, 32, 64, 128]

    for num_hidden in num_hidden_list:
        print("Number of hidden neurons:", num_hidden)
        model = cnn_model(num_hidden)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(
            x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test)
        )

        training_acc = history.history["accuracy"]
        validation_acc = history.history["val_accuracy"]

        plt.plot(training_acc, label="Training", linestyle="--", linewidth=3)
        plt.plot(validation_acc, label="Validation", linewidth=3)
        plt.title(
            "Accuracy for " + str(num_hidden) + " hidden neurons, kernel size 4",
            fontsize=14,
        )
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()


if __name__ == "__main__":
    draw_each()
