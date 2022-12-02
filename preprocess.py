import os
import random
from itertools import cycle

import cv2
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def load_data(image_dir):
    flower_list = ['Calla', 'Chrysanthemum', 'Hydrangea', 'Lavender', 'Rosa', 'PrunusMume', 'Orchid', 'Poppy', 'Sunflower']
    # flower_list = ['Calla', 'Chrysanthemum', 'Hydrangea', 'Lavender', 'Orchid',
    #                'Poppy', 'PrunusMume', 'Rosa', 'Sunflower']
    img_label_list = []
    for i in range(len(flower_list)):
        for file in os.listdir(image_dir + '/' + flower_list[i]):
            # if len(file.split('.')) > 2 and file.split('.')[2] == 'jpg':
            if file.split('.')[0] == flower_list[i]:
                img_label_list.append([image_dir + '/' + flower_list[i] + '/' + file, i])
    random.seed(0)  # Ensure that the data sequence is consistent each time
    random.shuffle(img_label_list)  # Scrambles all file paths

    image = []
    label = []
    for i in range(len(img_label_list)):
        img = cv2.imread(img_label_list[i][0])
        img = cv2.resize(img, (120, 120))  # Uniform picture size
        from keras.utils import img_to_array
        img = img_to_array(img)  # image to array
        image.append(img)
        label.append(img_label_list[i][1])
    image = np.array(image, dtype="float") / 255.0  # Normalization
    label = np.array(label)
    from keras.utils import to_categorical
    label = to_categorical(label, num_classes=len(flower_list))
    num_classes = len(flower_list)
    return image, label, num_classes


def spilt_train_test(image, label, ratio):
    train_image, test_image = train_test_split(image, test_size=ratio, random_state=42)
    train_label, test_label = train_test_split(label, test_size=ratio, random_state=42)
    return train_image, train_label, test_image, test_label


def test_accuracy(test_image, test_label, model):
    prediction = model.predict(test_image)
    result_list = []
    for i in prediction:
        result = np.zeros(i.shape)
        i = list(i)
        result[i.index(max(i))] = 1
        result_list.append(result)
    result_list = np.array(result_list)
    print(classification_report(np.argmax(test_label, axis=1), np.argmax(prediction, axis=1)))
    print(confusion_matrix(np.argmax(test_label, axis=1), np.argmax(prediction, axis=1)))

    accurate_account = 0
    for i in range(test_label.shape[0]):
        if list(result_list[i]) == list(test_label[i]):
            accurate_account += 1
    precision = accurate_account / test_label.shape[0]
    print(precision)
    return precision


def plot_roc(x_test, y_test, model, title):
    # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    classes = ['Calla', 'Chrysanthemum', 'Hydrangea', 'Lavender', 'Orchid',
               'Poppy', 'PrunusMume', 'Rosa', 'Sunflower']
    y_test = label_binarize(y_test, classes=classes)
    y_pre = model.predict(x_test)
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
    plt.title('ROC for multi-class classification - ' + title)
    plt.legend(loc="lower right")
    plt.show()


def baseline(x_train, y_train, x_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)

    pred_dummy = dummy.predict(x_train)
    y_pred_dummy = np.argmax(pred_dummy, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred_dummy))
    print(confusion_matrix(y_train1, y_pred_dummy))

    pred_dummy = dummy.predict(x_test)
    y_pred_dummy = np.argmax(pred_dummy, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred_dummy))
    print(confusion_matrix(y_test1, y_pred_dummy))

    title = 'Baseline'
    plot_roc(x_test, y_test, dummy, title)


def rotate():
    count = 1359
    for file in os.listdir('./images/Sunflower'):
        if file.split('.')[0] == 'Sunflower':
            count = count + 1
            print(file)
            im = Image.open('./images/Sunflower/' + file)
            # Specifies the Angle of counterclockwise rotation
            im_rotate = im.rotate(180)
            if im_rotate.mode != 'RGB':
                im_rotate = im_rotate.convert('RGB')
            im_rotate.save('./images/Sunflower/' + 'Sunflower.' + str(count) + '.jpg')
