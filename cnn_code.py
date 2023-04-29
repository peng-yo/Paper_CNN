from threading import main_thread

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import auc, recall_score, roc_curve
from tensorflow import keras

num_classes = 10
TOTAL_TEST_ACC = []
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# def cnn_model(num_hidden):
#     model = keras.models.Sequential()
#     model.add(
#         keras.layers.Conv2D(
#             num_hidden,
#             kernel_size=4,
#             padding="same",
#             activation="relu",
#             input_shape=(28, 28, 1),
#         )
#     )
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(num_hidden, activation="relu"))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(10, activation="softmax"))
#     return model


def cnn_model(num_hidden, kernel_size):
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(kernel_size, kernel_size),
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(Conv2D(num_hidden, (kernel_size, kernel_size), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def draw_each():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    num_hidden_list = [64, 128]
    # num_hidden_list = [64, 128]
    for kernel_size in [5]:
        for num_hidden in num_hidden_list:
            print("Number of hidden neurons:", num_hidden)
            model = cnn_model(num_hidden, kernel_size)
            model.compile(
                loss=keras.losses.categorical_crossentropy,
                # optimizer=keras.optimizers.legacy.Adadelta(),
                optimizer=keras.optimizers.legacy.Adam(),
                metrics=["accuracy"],
            )
            history = model.fit(
                x_train,
                y_train,
                epochs=10,
                batch_size=64,
                validation_data=(x_test, y_test),
            )
            # acc_list = [0 for i in range(10)]
            acc_list = []
            for i in range(10):
                model.fit(
                    x_train,
                    y_train,
                    epochs=1,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                )
                _, test_acc = model.evaluate(x_test, y_test)
                acc_list.append(test_acc)
                print(f"测试准确率: {acc_list}")
            TOTAL_TEST_ACC.append((num_hidden, acc_list[9]))
            training_acc = history.history["accuracy"]
            validation_acc = history.history["val_accuracy"]

            # write to file
            with open("acc.txt", "a") as f:
                f.write(
                    f"size {kernel_size}, num_hidden {num_hidden}:\ntest_acc:{acc_list}\n, train_acc:{training_acc}\n, validate_acc:{validation_acc} "
                )
            y_pred_prob = model.predict(x_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test, axis=1)

            recall = recall_score(y_true, y_pred, average="macro")
            with open("recall.txt", "a") as f:
                f.write(
                    f"kernel size: {kernel_size}, num_hidden:{num_hidden}召回率: {recall}"
                )
            # fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
            # roc_auc = auc(fpr, tpr)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            num_classes = y_train.shape[1]
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # plt.subplot(1, 2, 1)
            plt.plot(training_acc, label="Training", linestyle="--", linewidth=3)
            plt.plot(validation_acc, label="Validation", linewidth=3)
            plt.plot(acc_list, label="Testing", linestyle=":", linewidth=3)

            plt.title(
                "Accuracy for "
                + str(num_hidden)
                + f" hidden neurons, kernel size {kernel_size}",
                fontsize=14,
            )
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc="lower right", fontsize=12)
            plt.show()
            # plt.subplot(1, 2, 2)
            # # 预测概率
            # y_score = model.predict(x_test)

            # # 计算ROC曲线和AUC值
            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            # for i in range(num_classes):
            #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            #     roc_auc[i] = auc(fpr[i], tpr[i])

            # # 绘制ROC曲线
            # # plt.figure()
            # colors = [
            #     "aqua",
            #     "darkorange",
            #     "cornflowerblue",
            #     "green",
            #     "red",
            #     "purple",
            #     "pink",
            #     "gray",
            #     "brown",
            #     "olive",
            # ]
            # for i, color in zip(range(num_classes), colors):
            #     plt.plot(
            #         fpr[i],
            #         tpr[i],
            #         color=color,
            #         lw=2,
            #         label="ROC curve of class {0} (area = {1:0.2f})".format(
            #             i, roc_auc[i]
            #         ),
            #     )
            # plt.plot([0, 1], [0, 1], "k--", lw=2)
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel("False Positive Rate")
            # plt.ylabel("True Positive Rate")
            # plt.title(
            #     f"Receiver operating characteristic for multi-class, {num_hidden} hidden layers, kernel size {kernel_size}"
            # )
            # plt.legend(loc="lower right")
            # plt.show()


def draw_test_acc():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # kernel_3 = [0.9914000034332275, 0.9915, 0.9921, 0.9916]
    # kernel_5 = [0.9934, 0.9919, 0.9942, 0.9922]
    # # 将元组列表转换为 DataFrame
    x = np.array([16, 32, 64, 128])
    y1 = np.array([0.9922, 0.9928, 0.9929, 0.993])
    y2 = np.array([0.9943, 0.9944, 0.9942, 0.9935])
    data = np.stack((y1, y2), axis=1)
    df = pd.DataFrame(data, columns=["Kernel size 3", "Kernel size 5"])
    df["Number of Hidden Layers"] = x
    df = df.melt(
        "Number of Hidden Layers", var_name="Kernel size", value_name="Accuracy"
    )

    sns.lineplot(x="Number of Hidden Layers", y="Accuracy", hue="Kernel size", data=df)
    plt.title("Comparison of Kernel Size 3 and Kernel Size 5")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # draw_each()
    draw_test_acc()
