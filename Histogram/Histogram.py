#Histogram 

import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktarma
img = cv2.imread("red_blue.jpg") # resmi içe aktarıyoruz
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # remk düzenini ayarlıyoruz
plt.figure(), plt.imshow(img_vis) # resmi görselleştiriyoruz

print(img.shape) 

img_hist = cv2.calcHist([img], channels = [0],mask = None, histSize = [256], ranges = [0,256])  #cv2.calcHist görüntünün histogramını hesaplar. channels, histogram kanalını belirtir. mask, maskeleme kullanılacaksa değer verilir.
print(img_hist.shape)
plt.figure(), plt.plot(img_hist) #plt.plot, verileri grafik şeklinde görselleştirir.

color = ("b", "g", "r") # Çicgi grafiklerinde kullanılacak renkler belirtilmiştir.
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i],mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist, color = c)
    
    

golden_gate = cv2.imread("goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(golden_gate_vis)

print(golden_gate.shape)

mask = np.zeros(golden_gate.shape[:2], np.uint8)   
plt.figure(), plt.imshow(mask, cmap = "gray") #Görünütüye uygun maske oluşturuyoruz.

mask[1500:2000, 1000:2000] = 255 # Belirli bir alanı beyaz yapmak için kullanıyoruz. Bu sayede belirtilen koordinatların içi maskelenmez.
plt.figure(), plt.imshow(mask, cmap = "gray")

maskes_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask = mask)
plt.figure(), plt.imshow(maskes_img_vis, cmap = "gray")
    
masked_img = cv2.bitwise_and(golden_gate, golden_gate, mask = mask) # bitwise ile birleştirme yaparız. İlk matris değerleri, ikinci matris değerleri ve en sonda maske değerlerini alırız.
masked_img_hist =cv2.calcHist([golden_gate], channels = [0],mask = mask, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(masked_img_hist)
    

#Histogram Eşitleme
#Karşıtlık Arttırma
img = cv2.imread("hist_equ.jpg", 0)
plt.figure(), plt.imshow(img, cmap ="gray")

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize =[256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img) # Histogram eşitleme ile parlaklık dağılımını ayarlar ve görüntünün piksel yoğunluk değerini daha dengeli bir şekilde dağıtmayı amaçlar.
plt.figure(), plt.imshow(eq_hist, cmap ="gray")

eq_img_hist = cv2.calcHist([eq_hist], channels = [0], mask = None, histSize =[256], ranges = [0,256])
plt.figure(), plt.plot(eq_img_hist)