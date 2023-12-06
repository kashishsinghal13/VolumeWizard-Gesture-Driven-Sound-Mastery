import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

# Previous treble status
prev_treble_status = None

# Function to check if the index finger is pointing up (as an example for treble)
def detect_treble_status(hand_landmarks):
    # Check if all landmarks for the hand are present
    whole_hand_present = all(lm.HasField('x') for lm in hand_landmarks)

    # If the whole hand is present, check the index finger position
    if whole_hand_present:
        # Get the y-coordinate of the index finger tip
        index_finger_tip_y = hand_landmarks[8].y

        # Determine treble status based on the y-coordinate (adjust the threshold as needed)
        if index_finger_tip_y < hand_landmarks[6].y:  # Compare with the base of the index finger
            return "treble_up"
        else:
            return "treble_down"
    else:
        return "other"

# Your treble adjustment function (replace print statements with actual audio adjustment logic)
def adjust_treble(treble_status):
    global prev_treble_status

    # Implement your treble adjustment logic here
    # Example: Increase treble if the index finger is up
    if treble_status == "treble_up":
        print("Treble up. Increasing treble.")
        # Implement treble increase logic
    elif treble_status == "treble_down":
        print("Treble down. Decreasing treble.")
        # Implement treble decrease logic

    # Update the previous treble status
    prev_treble_status = treble_status

# Capture video from your webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get the treble status based on landmarks
            treble_status = detect_treble_status(landmarks.landmark)

            # Draw hand landmarks on the frame
            mpDraw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Adjust treble based on hand gesture
            adjust_treble(treble_status)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
