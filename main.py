import glob
import numpy as np
import pickle
#from functions import vision, support
from functions.support import *
from functions.vision import *
from ML.svm import *
from solver import pipeline

def testing(**kwargs):


    dict_data = kwargs.get('dict_samples')
    dict_labels = kwargs.get('dict_labels')

    train_data, train_labels = get_train_images()
    test_data, test_labels = get_test_images()
    train_hogs = []
    test_hogs = []

    for img in train_data:
        img = erosion(img, 3, 1)
        train_hogs.append(HOG(img))

    train_labels = train_labels.tolist()
    for index, img in enumerate(dict_data):
        train_hogs.append(img)
        train_labels.append(dict_labels[index])

    train_labels = np.asarray(train_labels)

    svm = svm_train(tr_data=train_hogs, tr_labels=train_labels, save=True)


    for img in test_data:
        img = erosion(img, 3, 1)
        #show_image(img)
        test_hogs.append(HOG(img))

    svm_classify(test_hogs, test_labels)


if __name__ == '__main__':

    #testing()
    #input("aa")

    #Store images in numpy array
    images = np.asarray(get_images(1))

    pipeline(images[0])

    #Resize and make grayscale
    img1 = resize(images[0], 900)
    gray_img = gray(img1)

    #Blur
    blur = gaussian_blur(gray_img, 15)
    thres = adaptive_thres(blur, False)

    #Get contours from thersholded image
    contours = get_contours(thres)

    #Get grid contour
    cnt, index = grid_contour(contours)

    #Get grid corner coordinate
    corner_points = corner_coordinates(cnt)

    #Make perpective transform
    img_corner_points = get_image_cornes(img1)
    transformed = perspective_transform(thres, corner_points, img_corner_points)

    grid = make_grid(transformed)
    number_samples = np.load("number_samples")
    hogs = {}
    for i in range(1,10):
        img = number_samples[str(i)]
        img = erosion(img, 3, 2)
        hogs[str(i)] = HOG(img)

    #tr_labels, tr_data = prepare_samples(hogs, 10)
    #test_data, test_labels = prepare_samples(hogs,5)
    tr_data, tr_labels = get_train_images()
    tr_hogs = []
    for tr_image in tr_data:
        #tr_image = erosion(tr_image, 3, 1)
        tr_hogs.append(HOG(tr_image))

    tr_labels = np.asarray(tr_labels)

    clf = svm_train(tr_data=tr_hogs, tr_labels=np.asarray(tr_labels), save=True)

    #testing(dict_samples=test_data, dict_labels=test_labels)
    #svm_train(tr_data=tr_data, tr_labels=tr_labels, save=False)
    #svm_classify(test_data, test_labels)




    """
    show_image(transformed)
    for column in grid:
        for img in column:
            if len(img) != 0:
                show_image(img[0])
                number = input("number: ")
    """