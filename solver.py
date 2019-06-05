from functions.support import *
from functions.vision import *
from ML.svm import *

def pipeline(img):

    img = resize(img, 900)
    gray_img = gray(img)

    blur = gaussian_blur(gray_img, 15)
    thres = adaptive_thres(blur, False)

    contours = get_contours(thres)
    cnt, index = grid_contour(contours)

    corner_points = corner_coordinates(cnt)

    img_corner_points = get_image_cornes(img)
    transformed = perspective_transform(thres, corner_points, img_corner_points)

    grid = make_grid(transformed)
    numbers = []
    for column in grid:
        for number in column:
            if len(number) != 0:
                #number = erosion(number[0], 3, 1)
                numbers.append(HOG(number[0]))


    correct = [9, 7, 1, 3, 2, 9, 5, 2, 6, 1, 4, 5, 3, 3, 5, 3, 5, 6,
               9, 8, 1, 3, 4, 1, 9, 7, 8, 6]
    predicted = svm_classify(test_data=numbers, test_labels=correct)
    print(predicted)
    show_image(transformed)
    input("asd")
