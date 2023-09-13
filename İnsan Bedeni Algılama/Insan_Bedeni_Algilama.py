import cv2 as cv

img = cv.imread("body.jpg") #Görüntüyü yüklüyoruz.
body_cascade = cv.CascadeClassifier("fullbody.xml") #İnsan bedenini algılaması için cascade dosyamnızı yüklüyoruz.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Görüntüyü gri tona çeviriyoruz.
bodies = body_cascade.detectMultiScale(gray,1.8,2 )

for (x,y,w,h) in bodies: # dikdörtgen oluşturmak için for döngüsü oluşturuyoruz.
    cv.rectangle(img, (x,y),(x+w,y+h),(0,0,255),3) # koordinatları, rengi ve çerçeve kalınlığını belirtiyoruz.
cv.imshow("image",img) # resmi gösteriyoruz

cv.waitKey(0) #herhangi bir tuş ile pencereden çıkışı sağlıyoruz
cv.destroyAllWindows()