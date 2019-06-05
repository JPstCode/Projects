import cv2 as cv
import glob
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from functions.vision import show_image, erosion, get_train_images, get_test_images

def HOG(img):

    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(16*ang/(2*np.pi))
    bin_cells = bins[:14,:14], bins[14:,:14], bins[:14,14:], bins[14:,14:]
    mag_cells = mag[:14,:14], mag[14:,:14], mag[:14,14:], mag[14:,14:]
    hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def prepare_samples(samples_dict, no_samples):
    from sklearn.utils import shuffle

    data_list = []
    label_list = []
    for number, data in samples_dict.items():
        for i in range(0,no_samples):
            data_list.append(data)
            label_list.append(number)

    x, y = shuffle(data_list, label_list)
    return x, y

def svm_train(**kwargs):

    tr_data = kwargs.get('tr_data')
    tr_labels = kwargs.get('tr_labels')
    save = kwargs.get('save')

    C = []
    gamma = []
    for i in range(1,10):
        C.append(int(i)/(1+int(i)))

    for i in range(3,6):
        gamma.append(int(i))

    #parameters = {'kernel': ('linear', 'rbf'), 'C': C, 'gamma': gamma}
    #svc = svm.SVC(decision_function_shape='ovo')
    #clf = GridSearchCV(svc, parameters, cv=5, return_train_score=True)

    clf = svm.SVC(C=0.5, gamma=3, kernel='linear')
    clf.fit(tr_data, tr_labels)


    if save:
        import pickle
        with open('svm_classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)
        return clf
    else:
        return clf


def svm_classify(**kwargs):

    test_data = kwargs.get('test_data')
    test_labels = kwargs.get('test_labels')
    if kwargs.get('clf') is not None:
        clf = kwargs.get('clf')
        predicted = clf.predict(test_data)

    else:
        import pickle
        with open('svm_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
            predicted = clf.predict(test_data)

    if test_labels is not None:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(test_labels, predicted))

    return predicted


if __name__ == '__main__':

    images = np.load('all_images.pkl')
    labels = np.load('all_image_labels.pkl')

    hogs = []
    for img in images:
        img = erosion(img, 3, 1)
        hogs.append(HOG(img))

    clf = svm_train(tr_data= hogs, tr_labels= labels, save= True)
