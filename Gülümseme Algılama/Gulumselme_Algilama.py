import cv2

img = cv2.imread("smile.jpg")
smile_cascade = cv2.CascadeClassifier("smile.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smiles = smile_cascade.detectMultiScale(img,5.0,3)

for (x,y,w,h) in smiles:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

cv2.imshow("Smile",img)

cv2.waitKey(0)
cv2.destroyAllWindows()