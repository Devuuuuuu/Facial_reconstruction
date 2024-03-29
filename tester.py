import cv2
import dlib

# Load the predictor for facial landmark detection
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the detector
detector = dlib.get_frontal_face_detector()

# Read the image
img = cv2.imread("face_img.png")  # Adjust the size as needed

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces using dlib's face detector
faces = detector(gray)

for face in faces:
    # Detect facial landmarks using dlib
    landmarks = predictor(gray, face)

    # Draw facial landmarks
    for n in range(0, 68):
        x_landmark = landmarks.part(n).x
        y_landmark = landmarks.part(n).y
        cv2.circle(img, (x_landmark, y_landmark), 3, (0, 255, 0), -1)

img = cv2.resize(img, (1024, 768))
# Display the image with facial landmarks
cv2.imshow("Facial Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
