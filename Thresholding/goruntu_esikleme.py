import cv2 
import matplotlib.pyplot as plt

#resmi içe aktar
img = cv2.imread("img1.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img, cmap = "gray")
plt.axis("off") #eksenleri kapatıyoruz
plt.show()

#eşikleme

_, thresh_img= cv2.threshold(img, thresh=60, maxval = 255, type=cv2.THRESH_BINARY_INV)
#cv2.threshold() fonksiyonunu kullanarak eşikleme işlemi yapılıyor. 
#Gri tonlamalı görüntünün piksel değerleri belirtilen eşik değeri (60) ile karşılaştırılıyor. 
#Eğer bir piksel değeri eşik değerinden büyükse, o piksel değeri 255 (beyaz) olarak kabul edilir, aksi halde 0 (siyah) olarak kabul edilir. 
#type=cv2.THRESH_BINARY_INV ise tersine çevrilen (inverted) eşikleme işlemi yapılacağını belirtir. 
#Elde edilen eşiklenmiş görüntüyü thresh_img adlı değişkende saklarsınız.
plt.figure()
plt.imshow(thresh_img, cmap = "gray")
plt.axis("off")
plt.show() 

#uyarlamalı eşik değeri

thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,8)
#Bu satırda, adaptif eşikleme işlemi yapılıyor. 
#cv2.adaptiveThreshold() fonksiyonu ile görüntünün lokal alanlarında adaptif bir eşik değeri uygulanır. 
#cv2.ADAPTIVE_THRESH_MEAN_C metodu ile adaptif eşik değeri hesaplanır. 
#cv2.THRESH_BINARY ise eşikleme işleminin yapılacağını ve eşik değeri üzerinde veya altında olan piksellerin beyaz (255), üstünde olanların siyah (0) olarak kabul edileceğini belirtir. 
#11 adaptif eşik değerini hesaplamak için kullanılan blok boyutunu belirtirken, 8 ise hesaplanan eşik değerine eklenen sabiti ifade eder.
plt.figure()
plt.imshow(thresh_img2, cmap ="gray")
plt.axis("off")
plt.show()