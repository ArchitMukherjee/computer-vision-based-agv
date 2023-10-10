import cv2 as cv

cap = cv.VideoCapture(2)
num = 0
#cap = cv.VideoCapture('tarp_test_video_calib.mp4')

while cap.isOpened():

    succes, img = cap.read()

    k = cv.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite('cameraCaliberation/images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv.destroyAllWindows()
