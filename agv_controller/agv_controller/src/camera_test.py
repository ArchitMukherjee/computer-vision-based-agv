import cv2 as cv

cap = cv.VideoCapture(0)


cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()
    img = img[:, 50:800]
    h, w, _ = img.shape
    print(h,w)
    
    #print (x,y,rvec1)
    cv.imshow("Video", img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
