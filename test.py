import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands
hands = mp_hands.Hands()

# Open the camera
cap = cv2.VideoCapture(0)

# Set initial volume level
initial_volume = pyautogui.getVolume()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract thumb and index finger landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the vertical distance between thumb and index finger
            distance = thumb_tip.y - index_tip.y

            # Adjust the volume based on the distance
            new_volume = initial_volume + (distance * 100)
            pyautogui.press('volumeup') if new_volume > initial_volume else pyautogui.press('volumedown')

    # Draw hand landmarks on the frame
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
