import cv2

cat_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
)

video_capture = cv2.VideoCapture(0)

if video_capture.isOpened():
    ret, frame = video_capture.read()
else:
    rt = False

while ret:
    # cv2.imshow("Video ", frame)

    ret, frame = video_capture.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

    # แปลงภาพเป็นสี
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = cat_cascade.detectMultiScale(gray, 1.1, 3)

    print("Found {0} cat faces!".format(len(faces)))

    # วาดสี่เหลี่ยมรอบใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 4)  # สีเขียว ความหนา 4
        cv2.putText(frame, 'cat face', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)  # ข้อความสีเขียว ควาวหนา 1

    # แสดงภาพผลลัพย์
    cv2.imshow('Cat face detection', frame)

video_capture.destroyWindow("Video")
video_capture.release()
