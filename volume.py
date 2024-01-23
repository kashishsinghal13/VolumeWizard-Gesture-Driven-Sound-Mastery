# import cv2
# import mediapipe as mp
# from math import hypot
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# import numpy as np

# def volu():
#     cap = cv2.VideoCapture(0)  # Checks for camera

#     mpHands = mp.solutions.hands  # detects hand/finger
#     hands = mpHands.Hands()  # complete the initialization configuration of hands
#     mpDraw = mp.solutions.drawing_utils

#     # To access speaker through the library pycaw
#     devices = AudioUtilities.GetSpeakers()
#     interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#     volume = cast(interface, POINTER(IAudioEndpointVolume))
#     volbar = 400
#     volper = 0

#     volMin, volMax = volume.GetVolumeRange()[:2]

#     while True:
#         success, img = cap.read()  # If camera works capture an image
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to rgb

#         # Collection of gesture information
#         results = hands.process(imgRGB)  # completes the image processing.

#         lmList = []  # empty list
#         if results.multi_hand_landmarks:  # list of all hands detected.
#             # By accessing the list, we can get the information of each hand's corresponding flag bit
#             for handlandmark in results.multi_hand_landmarks:
#                 for id, lm in enumerate(handlandmark.landmark):  # adding counter and returning it
#                     # Get finger joint points
#                     h, w, _ = img.shape
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
#                 mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

#         if lmList != []:
#             # getting the value at a point
#             # x      #y
#             x1, y1 = lmList[4][1], lmList[4][2]  # thumb
#             x2, y2 = lmList[8][1], lmList[8][2]  # index finger
#             # creating circle at the tips of thumb and index finger
#             cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
#             cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
#             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # create a line b/w tips of index finger and thumb

#             length = hypot(x2 - x1, y2 - y1)  # distance b/w tips using hypotenuse
#             # from numpy we find our length,by converting hand range in terms of volume range ie b/w -63.5 to 0
#             vol = np.interp(length, [30, 350], [volMin, volMax])
#             volbar = np.interp(length, [30, 350], [400, 150])
#             volper = np.interp(length, [30, 350], [0, 100])

#             print(vol) #int(length)
#             volume.SetMasterVolumeLevel(vol, None)

#             # Hand range 30 - 350
#             # Volume range -63.5 - 0.0
#             # creating volume bar for volume level
#             cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255),
#                           4)  # vid ,initial position ,ending position ,rgb ,thickness
#             cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
#             cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
#             # tell the volume percentage ,location,font of text,length,rgb color,thickness
#         cv2.imshow('Image', img)  # Show the video
#         if cv2.waitKey(1) & 0xff == ord(' '):  # By using spacebar delay will stop
#             break

#     cap.release()  # stop cam
#     cv2.destroyAllWindows()


# volu()# close window

import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

# Function to control audio based on hand gestures
def control_audio():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_bar = 400
    vol_min, vol_max = volume.GetVolumeRange()[:2]

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Error: Empty frame.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        lm_list = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list.extend([(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0]))
                                for id, lm in enumerate(hand_landmark.landmark)])

                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            x1, y1 = lm_list[4][1:3]
            x2, y2 = lm_list[8][1:3]

            cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Control volume based on hand gestures
            length = hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [30, 350], [vol_min, vol_max])
            vol_bar = np.interp(length, [30, 350], [400, 150])

            print(vol, int(length))
            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f"{int(np.interp(length, [30, 350], [0, 100]))}%",
                        (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

        cv2.imshow('Hand Gestures', img)

        if cv2.waitKey(1) & 0xff == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Controlling Audio based on Hand Gestures...")
control_audio()
