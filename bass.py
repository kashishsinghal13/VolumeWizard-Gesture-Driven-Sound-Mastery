import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

# Previous thumb status
prev_thumb_status = None

# Function to check if the thumb is up
def detect_thumb_status(hand_landmarks):
    # Check if all landmarks for the hand are present
    whole_hand_present = all(lm.HasField('x') for lm in hand_landmarks)

    # If the whole hand is present, check the thumb position
    if whole_hand_present:
        # Get the y-coordinate of the thumb tip
        thumb_tip_y = hand_landmarks[4].y

        # Determine thumb status based on the y-coordinate (adjust the threshold as needed)
        if thumb_tip_y < hand_landmarks[2].y:  # Compare with the base of the thumb
            return "thumb_up"
        else:
            return "thumb_down"
    else:
        return "other"

# Your bass adjustment function
def adjust_bass(thumb_status):
    global prev_thumb_status

    # Implement your bass adjustment logic here
    # Example: Increase bass if the thumb is up
    if thumb_status == "thumb_up" :
        print("Thumb up. Increasing bass.")
        # Implement bass increase logic
    elif thumb_status == "thumb_down" :
        print("Thumb down. Decreasing bass.")
        # Implement bass decrease logic

    # Update the previous thumb status
    prev_thumb_status = thumb_status

# Capture video from your webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Get the thumb status based on landmarks
                thumb_status = detect_thumb_status(landmarks.landmark)

                # Draw hand landmarks on the frame
                mpDraw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Adjust bass based on thumb status
                adjust_bass(thumb_status)
    except Exception as e:
        print(f"Error in processing frame: {e}")

    cv2.imshow('Bass Control', frame)

    if cv2.waitKey(1) & 0xff == ord(' '):  # By using spacebar delay will stop
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()


