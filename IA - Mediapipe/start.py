import cv2
import mediapipe as mp

# Capture du flux vidéo de la webcam
cap = cv2.VideoCapture(0)

# Detection des mains
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read() # Lecture frame par frame du flux vidéo
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Conversion de la frame
    results = hands.process(imgRGB) # Detection des mains sur la frame

    if results.multi_hand_landmarks: # Si main detecter
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img) # Affiche l'image avec les traits de la main
    cv2.waitKey(1)
