import cv2
import streamlit as st
import tempfile
import os
import time

# initialize the webcam
cap = cv2.VideoCapture(0)

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# funtion to detect faces
def detech_face(img):
    # แปลงภาพเป็นสีgmk
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    # วาดสี่เหลี่ยมรอบใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (0, 255, 0), 4)  # สีเขียว ความหนา 4
    return img


st.title("Face Detection")
run = st.checkbox("Run")

FRAME_WINDOW = st.image([])


while run:
    ret, frame = cap.read()
    if not ret:
        continue
    # Detect faces
    frame = detech_face(frame)

    # Display frame to browser windows
    FRAME_WINDOW.image(frame, channels="BGR")

    # save image if button is click
    if st.button('save image'):
        timestamp = int(time.time())
        filename = f"saved_image_{timestamp}.jpg"

        cv2.imwrite(f"images/{filename}", frame)

cap.release()
