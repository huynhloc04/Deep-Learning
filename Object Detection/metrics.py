
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def confusion_mtr(y_true, y_pred):
    N = np.unique(y_pred).shape[0]
    cm = np.zeros((N, N), dtype = np.uint8) 
    for i in range(y_pred.shape[0]):
        cm[y_true[i], y_pred[i]] += 1
    return cm



def plot_confusion_matrix(cm, classes, 
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / np.sum(cm, axis=1, keepdims = True)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, format(cm[i][j], fmt), 
                 horizontalalignment="center",
                 color = 'white' if cm[i][j] > thresh else 'black')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.show()


y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 1, 2])
conf_mtr = confusion_mtr(y_true, y_pred)


class_names = [0, 1, 2]
plot_confusion_matrix(conf_mtr, normalize = True, classes=class_names, title = 'Confusion matrix, without normalization')
