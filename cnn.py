import numpy as np
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import keras
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt

from preprocess import load_data, spilt_train_test, test_accuracy, plot_roc, baseline


def cnn(x_train, y_train, num_classes, batch_size, epochs):
    model = keras.Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=x_train.shape[1:], name="conv1"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu", name="conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))
    # model.add(Conv2D(80, (5, 5), padding="same", activation="relu", name="conv3"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool3"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation="relu", name="fc1"))
    model.add(Dense(num_classes, activation="softmax", name="fc2"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    # model.save("cnn.model")
    keras.models.save_model(model, 'cnn4.model')

    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    return model


file_dir = './images'
images, labels, num_of_classes = load_data(file_dir)
train_image, train_label, test_image, test_label = spilt_train_test(images, labels, 0.2)

model = cnn(train_image, train_label, num_of_classes, 64, 15)

model = keras.models.load_model('cnn4.model')


test_accuracy(test_image, test_label, model)

baseline(train_image, train_label, test_image, test_label)

plot_roc(test_image, test_label, model, 'CNN')

