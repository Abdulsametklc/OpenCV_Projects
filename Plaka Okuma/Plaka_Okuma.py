#Resmi gri formata çeviriyoruz.  cv2.cvtColor()
#Köşeleri yumuşatıyoruz. cv2.bilateralFilter()
#Resimdeki köşeleri test ediyoruz. cv2.Canny()
#Plakanın konturlarını elde ediyoruz. cv2.findContours()
#Maske uygulayıp kırptıktan sonra, plakaya okuyoruz.

import cv2
import numpy as np
import pytesseract
import imutils

img = cv2.imread("licence_plate.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray,6,250,250) #İlk gri tonlu resmi giriyoruz.
edged = cv2.Canny(filtered,30,200)


contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(contours) #Konturları yakalıyoruz.
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] #sorted ile sıralıyoruz. İkinci parametre alana göre sıralama. Üçüncü parametre girilen değerleri tersten sırala.
#Sondaki [:10] 0-10'a kadar olan değerler için yap
screen = None

for c in cnts:
    epsilon = 0.018*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    if len(approx) ==4:
        screen = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask,[screen],0,(255,255,255),-1)
new_img = cv2.bitwise_and(img,img, mask=mask)

(x,y) = np.where(mask ==255)
(topx,topy) = (np.min(x),np.min(y))
(bottomx,bottomy) =(np.max(x),np.max(y))
cropped = gray[topx:bottomx+1,topy:bottomy+1]

text = pytesseract.image_to_string(cropped, lang="eng")
print("detected text: ",text)

cv2.waitKey(0)
cv2.destroyAllWindows()
