import cv2

video = cv2.VideoCapture(0)
smile_cascade = cv2.CascadeClassifier("frontal_cascade.xml")

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (640,480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    smiles = smile_cascade.detectMultiScale(frame, 1.5,2)

    for (x,y,w,h) in smiles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("Smile",frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()