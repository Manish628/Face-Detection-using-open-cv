import cv2

# Correct way to load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(r"C:\Users\singh\open cv project\face detection\haarcascade_frontalface_default.xml")


# Check if the cascade file is loaded correctly
if face_cascade.empty():
    print("Error loading cascade classifier!")
    exit()

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, img = webcam.read()
    if not ret:
        break  # Exit if frame not captured

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show the output
    cv2.imshow("Face Detection", img)

    # Exit on pressing 'ESC' (key code 27)
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
