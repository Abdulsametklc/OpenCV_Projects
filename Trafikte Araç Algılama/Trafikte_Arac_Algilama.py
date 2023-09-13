import cv2

img = cv2.imread("car.jpg") #Fotoğrafı içe aktarıyoruz.
car_cascade = cv2.CascadeClassifier("car.xml") #Cascade dosyasını ekliyoruz.

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Resmi gri ton yapıyoruz.
car = car_cascade.detectMultiScale(img, 1.1,2) #Tespit edilecek nesneleri belirliyoruz.

for (x,y,w,h) in car: #Dikdörtgen oluşturuyoruz.
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),3) #Koordinat, Çerçeve Rengi, Kalınlığı
cv2.imshow("image",img) #Görüntüleme

cv2.waitKey(0) #Serbest Bırakma
cv2.destroyAllWindows()