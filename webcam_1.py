import numpy as np
import cv2 as cv

def nothing(x):
    pass


cap = cv.VideoCapture(0)
#cv.namedWindow('Trackbar',cv.WINDOW_AUTOSIZE)

cv.createTrackbar('B', 'Trackbar',0,255,nothing)
cv.createTrackbar('W', 'Trackbar',0,255,nothing)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    B = cv.getTrackbarPos('B','Trackbar')
    W = cv.getTrackbarPos('W','Trackbar')

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (2, 2))

    _, thres = cv.threshold(blur, 170, 255, cv.THRESH_BINARY_INV)


#    cnt = contours(thres,frame)
    _, contours, _ = cv.findContours(thres, cv.RETR_TREE,
                                     cv.CHAIN_APPROX_SIMPLE)

    lines = cv.HoughLinesP(thres, 1, np.pi / 180, 100, 150, 150)
    #print(lines.shape)
    leftmost = 0
    rightmost = 0
    topmost = 0
    bottommost = 0

    for i in range(0,len(contours)):
        if cv.contourArea(contours[i]) > 75000 and \
                cv.contourArea(contours[i]) < 100000 and \
                len(lines) < 100:
            #print(lines.shape)
            cv.drawContours(frame, contours, i, (0, 255, 0), 2)
            area = cv.contourArea(contours[i])
            cnt = contours[i]
            #print(area)
            #print(np.sum(thres)/100)
            #epsilon = 0.1 * cv.arcLength(cnt, True)
            #approx = cv.approxPolyDP(cnt, epsilon, True)
            #roi = cv.approxPolyDP(cnt,epsilon,True)

            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            w = np.abs(leftmost[0] - topmost[0])
            h = np.abs(leftmost[1] - topmost[1])


            #if len(roi) == 4:

            #    print(roi[0][0][0])
            #    print(roi)
                #input("asd")
                #cv.imshow('roi',frame[[roi[0][0][0]]:roi[0][1][0],roi[0][0][1]:roi[0][1][1]])
            #    cv.waitKey(0)
            #    cv.destroyAllWindows()


    # Display the resulting frame
    cv.imshow('frame',frame)
    cv.imshow('thres',thres)

    if np.sum(leftmost) != 0:
        print(leftmost)
        print(rightmost)
        print(topmost)
        print(bottommost)

        cv.imshow('roi',frame[leftmost[0]:rightmost[0],leftmost[1]:rightmost[1]])




    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()