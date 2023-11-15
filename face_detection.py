import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

if video_capture.isOpened():
    ret, frame = video_capture.read()
else:
    rt = False

while ret:
    cv2.imshow("Video ", frame)
    ret, frame = video_capture.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

    # แปลงภาพเป็นสี
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    face = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    # วาดสี่เหลี่ยมรอบใบหน้า
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 4)  # สีเขียว ความหนา 4


video_capture.destroyWindow("Video")
video_capture.release()
