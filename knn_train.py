from itertools import cycle

from numpy import interp
from sklearn.metrics import confusion_matrix, auc, classification_report
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import img_to_array
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2

import roc
from extract_features import *
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

num_color_bins = 10  # Set the bin size of the color histogram

# Set the two feature extractors that need to be called next, namely hog_feature, color_histogram_hsv
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]


def preprocess(image_dir):
    flower_list = ['Calla', 'Chrysanthemum', 'Hydrangea', 'Lavender', 'Rosa', 'PrunusMume', 'Orchid', 'Poppy',
                   'Sunflower']
    img_lable_list = []
    for i in range(len(flower_list)):
        for file in os.listdir(image_dir + '/' + flower_list[i]):
            if file.split('.')[0] == flower_list[i]:
                img_lable_list.append([image_dir + '/' + flower_list[i] + '/' + file, i])
    random.seed(0)  # Ensure that the data sequence is consistent every time
    random.shuffle(img_lable_list)  # shuffle all file paths

    data = []
    label = []
    print(len(img_lable_list))
    for i in range(len(img_lable_list)):
        img = cv2.imread(img_lable_list[i][0])
        img = cv2.resize(img, (220, 170))  # Uniform image size
        img = img_to_array(img)  # Image to array
        data.append(img)
        label.append(img_lable_list[i][1])
    data = np.array(data, dtype="float") / 255.0  # Normalized
    label = np.array(label)
    label = to_categorical(label, num_classes=len(flower_list))
    num_classes = len(flower_list)
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(label, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, num_classes


def knn(x_train, x_test, y_train, y_test, num_classes, batch_size, epochs):
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train = extract_features(x_train, feature_fns, verbose=True)
    X_test = extract_features(x_test, feature_fns)
    x_normalizer = StandardScaler()
    X_train = x_normalizer.fit_transform(X_train)
    X_test = x_normalizer.transform(X_test)
    params_k = [1, 3, 5, 7]  # K value that can be selected
    params_p = [1, 2]  # P-values that can be chosen
    param_grid = {"n_neighbors": params_k, "p": params_p}
    # build model
    clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    predict_model = clf.best_estimator_
    y_pre = predict_model.predict(X_test)
    print(accuracy_score(y_test, y_pre))
    y_test1 = np.argmax(y_test, axis=1)
    y_pre1 = np.argmax(y_pre, axis=1)
    print(confusion_matrix(y_test1, y_pre1))
    print(classification_report(y_test1, y_pre1))

    y_test = label_binarize(y_test, classes=classes)
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow', 'lime', 'crimson'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    count = 0
    for i in range(0, len(y_pre)):
        if list(y_pre[i]) == list(y_test[i]):
            count = count + 1
    print("Predict precision is: ", count / len(y_test))


file_dir = './images'
x_train, x_test, y_train, y_test, num_of_classes = preprocess(file_dir)
knn(x_train, x_test, y_train, y_test, num_of_classes, 10, 60)
