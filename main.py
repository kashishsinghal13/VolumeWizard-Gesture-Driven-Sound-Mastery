import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import speech_recognition as sr


def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not hear your command.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


def control_audio(mode):
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

            if mode == 'volume':
                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [30, 350], [vol_min, vol_max])
                vol_bar = np.interp(length, [30, 350], [400, 150])

                print(vol, int(length))
                volume.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f"{int(np.interp(length, [30, 350], [0, 100]))}%",
                            (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
            elif mode == 'bass':
                thumb_status = detect_thumb_status(lm_list)
                adjust_bass(thumb_status)

        cv2.imshow('Hand Gestures', img)

        if cv2.waitKey(1) & 0xff == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_thumb_status(hand_landmarks):
    whole_hand_present = all(isinstance(x, int) and isinstance(y, int) for _, x, y in hand_landmarks)

    if whole_hand_present:
        thumb_tip_y = hand_landmarks[4][2]
        return "thumb_up" if thumb_tip_y < hand_landmarks[2][2] else "thumb_down"
    else:
        return "other"


def adjust_bass(thumb_status):
    global prev_thumb_status

    if thumb_status == "thumb_up":
        print("Thumb up. Increasing bass.")
        # Implement bass increase logic
    elif thumb_status == "thumb_down":
        print("Thumb down. Decreasing bass.")
        # Implement bass decrease logic

    prev_thumb_status = thumb_status


if __name__ == "__main__":
    print("Listening for voice command...")

    while True:
        command = recognize_speech()

        if command:
            if "volume" in command:
                print("Controlling Volume...")
                control_audio('volume')
            elif "control" in command:
                print("Controlling Bass...")
                control_audio('bass')
            elif "exit" in command:
                print("Exiting...")
                break
            else:
                print("Command not recognized. Try saying 'volume' or 'bass'.")