import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

# Previous hand position
prev_hand_position = None

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

# Function to check hand position and adjust treble

def adjust_treble(hand_landmarks):
    global prev_hand_position

    # Get the tips of thumb and middle finger
    thumb_tip = (hand_landmarks[4].x, hand_landmarks[4].y)
    middle_finger_tip = (hand_landmarks[8].x, hand_landmarks[8].y)

    # Calculate the distance between thumb tip and middle finger tip
    hand_distance = calculate_distance(thumb_tip, middle_finger_tip)

    # Print additional debug information
    print(f"Thumb Tip: {thumb_tip}, Middle Finger Tip: {middle_finger_tip}")
    print(f"Hand Distance: {hand_distance}")

    # Adjust treble based on hand distance
    treble = int(np.interp(hand_distance, [50, 200], [0, 100]))

    # Print the treble value
    print(f"Treble: {treble}")

    # Determine hand position based on treble value
    hand_position = "near" if treble > 50 else "far"

    # Print the hand position
    print(f"Hand Position: {hand_position}")

    # Check if hand position has changed
    if prev_hand_position is not None and prev_hand_position != hand_position:
        # Print the change in treble
        treble_change = abs(treble - prev_hand_position)
        print(f"Treble Change: {treble_change}")

    # Update the previous hand position
    prev_hand_position = hand_position

    # Implement your treble adjustment logic here
    # Example: Adjust audio treble based on hand position
    # You can use the 'treble' variable to control the treble level in your audio system



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
                # Draw hand landmarks on the frame
                mpDraw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Adjust treble based on hand position
                adjust_treble(landmarks.landmark)
    except Exception as e:
        print(f"Error in processing frame: {e}")

    cv2.imshow('Treble Control', frame)

    if cv2.waitKey(1) & 0xff == ord(' '):  # By using spacebar, the loop will stop
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
