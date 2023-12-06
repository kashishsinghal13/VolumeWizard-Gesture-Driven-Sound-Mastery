#main.py
# import cv2
# import mediapipe as mp
# from math import hypot
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# import numpy as np
# import speech_recognition as sr

# # Global variables
# prev_treble_status = None
# prev_thumb_status = None

# # Function to recognize speech
# def recognize_speech():
#     recognizer = sr.Recognizer()

#     with sr.Microphone() as source:
#         print("Say something:")
#         recognizer.adjust_for_ambient_noise(source, duration=5)
#         audio = recognizer.listen(source, timeout=10)

#     try:
#         text = recognizer.recognize_google(audio).lower()
#         return text
#     except sr.UnknownValueError:
#         print("Sorry, I did not hear your command.")
#         return None
#     except sr.RequestError as e:
#         print(f"Could not request results from Google Speech Recognition service; {e}")
#         return None
#     except sr.WaitTimeoutError:
#         print("Listening timed out. No speech detected.")
#         return None

# # Function to detect thumb status
# def detect_thumb_status(hand_landmarks):
#     whole_hand_present = all(isinstance(x, int) and isinstance(y, int) for _, x, y in hand_landmarks)

#     if whole_hand_present:
#         thumb_tip_y = hand_landmarks[4][2]
#         return "thumb_up" if thumb_tip_y < hand_landmarks[2][2] else "thumb_down"
#     else:
#         return "other"

# # Function to adjust bass based on thumb status
# def adjust_bass(thumb_status):
#     global prev_thumb_status

#     if thumb_status == "thumb_up":
#         print("Thumb up. Increasing bass.")
#         # Implement bass increase logic
#     elif thumb_status == "thumb_down":
#         print("Thumb down. Decreasing bass.")
#         # Implement bass decrease logic

#     prev_thumb_status = thumb_status

# # Function to detect treble status
# # Function to detect treble status
# def detect_treble_status(hand_landmarks):
#     whole_hand_present = all(isinstance(x, int) and isinstance(y, int) for _, x, y in hand_landmarks)

#     if whole_hand_present:
#         index_finger_tip_y = hand_landmarks[8][2]  # Assuming the index finger is at index 8 in the landmarks list
#         base_of_index_finger_y = hand_landmarks[0][2]  # Assuming the wrist is at index 0 in the landmarks list

#         if index_finger_tip_y < base_of_index_finger_y:
#             return "treble_up"
#         else:
#             return "treble_down"
#     else:
#         return "other"



# # Function to adjust treble based on hand gesture
# def adjust_treble(treble_status):
#     global prev_treble_status

#     if treble_status == "treble_up":
#         print("Treble up. Increasing treble.")
#         # Implement treble increase logic
#     elif treble_status == "treble_down":
#         print("Treble down. Decreasing treble.")
#         # Implement treble decrease logic

#     prev_treble_status = treble_status

# # Function to control audio based on mode ('volume' or 'bass' or 'treble')
# def control_audio(mode):
#     cap = cv2.VideoCapture(0)

#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands()
#     mp_draw = mp.solutions.drawing_utils

#     devices = AudioUtilities.GetSpeakers()
#     interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#     volume = cast(interface, POINTER(IAudioEndpointVolume))
#     vol_bar = 400
#     vol_min, vol_max = volume.GetVolumeRange()[:2]

#     while True:
#         success, img = cap.read()
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)

#         lm_list = []
#         if results.multi_hand_landmarks:
#             for hand_landmark in results.multi_hand_landmarks:
#                 lm_list.extend([(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0]))
#                                 for id, lm in enumerate(hand_landmark.landmark)])

#                 mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

#         if lm_list:
#             x1, y1 = lm_list[4][1:3]
#             x2, y2 = lm_list[8][1:3]

#             cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
#             cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
#             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

#             if mode == 'volume':
#                 length = hypot(x2 - x1, y2 - y1)
#                 vol = np.interp(length, [30, 350], [vol_min, vol_max])
#                 vol_bar = np.interp(length, [30, 350], [400, 150])

#                 print(vol, int(length))
#                 volume.SetMasterVolumeLevel(vol, None)

#                 cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
#                 cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
#                 cv2.putText(img, f"{int(np.interp(length, [30, 350], [0, 100]))}%",
#                             (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
#             elif mode == 'bass':
#                 thumb_status = detect_thumb_status(lm_list)
#                 adjust_bass(thumb_status)
#             elif mode == 'treble':
#                 treble_status = detect_treble_status(lm_list)
#                 adjust_treble(treble_status)

#         cv2.imshow('Hand Gestures', img)

#         if cv2.waitKey(1) & 0xff == ord(' '):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     print("Listening for voice command...")

#     while True:
#         command = recognize_speech()

#         if command:
#             if "volume" in command:
#                 print("Controlling Volume...")
#                 control_audio('volume')
#             elif "control" in command:
#                 print("Controlling Bass...")
#                 control_audio('bass')
#             elif "no" in command:
#                 print("Controlling Treble...")
#                 control_audio('treble')
#             elif "exit" in command:
#                 print("Exiting...")
#                 break
#             else:
#                 print("Command not recognized. Try saying 'volume', 'bass', 'treble', or 'exit'.")

import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

def volu():
    cap = cv2.VideoCapture(0)  # Checks for the camera

    mpHands = mp.solutions.hands  # detects hand/finger
    hands = mpHands.Hands()  # complete the initialization configuration of hands
    mpDraw = mp.solutions.drawing_utils

    # To access the speaker through the library pycaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volMin, volMax = volume.GetVolumeRange()[:2]

    # Initialize volbar before the loop
    volbar = 400

    while True:
        success, img = cap.read()  # If the camera works, capture an image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Collection of gesture information
        results = hands.process(imgRGB)  # completes the image processing.

        lmList = []  # empty list
        if results.multi_hand_landmarks:  # list of all hands detected.
            # By accessing the list, we can get the information of each hand's corresponding flag bit
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):  # adding counter and returning it
                    # Get finger joint points
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        if lmList != []:
            x1, y1 = lmList[4][1], lmList[4][2]  # thumb
            x2, y2 = lmList[8][1], lmList[8][2]  # index finger

            # creating circle at the tips of thumb and index finger
            cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
            cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # create a line b/w tips of index finger and thumb

            length = hypot(x2 - x1, y2 - y1)  # distance b/w tips using hypotenuse

            # from numpy we find our length, by converting hand range in terms of volume range ie b/w -63.5 to 0
            vol = np.interp(length, [30, 350], [volMin, volMax])
            volbar = np.interp(length, [30, 350], [400, 150])
            volper = np.interp(length, [30, 350], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            # Display the bass level on the video feed
            cv2.putText(img, f"Bass: {int(vol)}", (10, 80), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

        # creating volume bar for volume level
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)  # vid ,initial position ,ending position ,rgb ,thickness
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
        # tell the volume percentage ,location,font of text,length,rgb color,thickness

        cv2.imshow('Image', img)  # Show the video
        if cv2.waitKey(1) & 0xff == ord(' '):  # By using spacebar delay will stop
            break

    cap.release()  # stop cam
    cv2.destroyAllWindows()

volu()  # close window
