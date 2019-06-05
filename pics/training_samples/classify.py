import cv2 as cv
import glob
import numpy as np
from functions.vision import show_image

images = [cv.imread(file,0) for file in glob.glob(
    r'C:\Users\juhop\Python_Files\Sudoku\test_samples\*.png')]

test_samples = []
index = 0
for image in images:
    show_image(image)
    value = int(input(str(index) + "number: "))
    test_samples.append(value)
    index = index + 1

np.save('test_samples',test_samples)

samples = np.load('test_samples.npy')
print(samples)