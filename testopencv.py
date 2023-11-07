import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a video capture object
vid = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = vid.read()

    # Convert color image to grayscale for Viola-Jones
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(grayscale_img, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# When everything is done, release the capture
vid.release()
cv2.destroyAllWindows()
