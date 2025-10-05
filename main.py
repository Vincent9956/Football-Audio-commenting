import cv2 as cv

img = cv.imread('Soldier.png')
cv.imshow('Soldier.jpg', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces = haar_cascade.detectMultiScale(gray, 1.1, 3)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow('face', img)
cv.waitKey(0)
cv.destroyAllWindows()


