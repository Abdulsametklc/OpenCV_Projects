import cv2

video = cv2.VideoCapture("body.mp4") # videoyu içe aktarıyoruz
body_cascade = cv2.CascadeClassifier("fullbody.xml")#cascade dosyasını yüklüyoruz

while True:
    ret, frame = video.read() # Tek tek frameleri okuyoruz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Frameleri griye çeviriyoruz.

    bodies = body_cascade.detectMultiScale(gray, 1.1, 4) #Gelen görüntüdeki istenenleri tespit ediyoruz. 1.1 ölçeklendirme, 4 iterasyon

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("video",frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()