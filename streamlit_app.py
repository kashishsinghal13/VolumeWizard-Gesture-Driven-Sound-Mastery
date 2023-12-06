import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import speech_recognition as sr
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Hand Gesture ", page_icon="🖐️", layout="centered")


st.write(
    f'<div style="display: flex; align-items: center; justify-content: space-between;">'
    f'<div><img src="https://cdn.dribbble.com/users/378514/screenshots/10081609/media/f1caa49981d0c2a25e462f302478e0fa.png?resize=400x300&vertical=center" width="100" alt="🖐️ Hand Gesture Logo"></div>'
    f'<div style="text-align: left; font-style: italic;">'
    f'<h1>VolumeWizard: Gesture-Driven Sound Mastery</h1>'
    f'</div></div>',
    unsafe_allow_html=True
)
# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Say something:")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            st.write("Timeout: No speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I did not hear your command.")
        return None
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Function to control audio based on hand gestures
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
        if not success or img is None:
            st.write("Error: Empty frame.")
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

            if mode == 'volume':
                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [30, 350], [vol_min, vol_max])
                vol_bar = np.interp(length, [30, 350], [400, 150])

                st.write(vol, int(length))
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

# Function to detect thumb status
def detect_thumb_status(hand_landmarks):
    whole_hand_present = all(isinstance(x, int) and isinstance(y, int) for _, x, y in hand_landmarks)

    if whole_hand_present:
        thumb_tip_y = hand_landmarks[4][2]
        return "thumb_up" if thumb_tip_y < hand_landmarks[2][2] else "thumb_down"
    else:
        return "other"

# Function to adjust bass
def adjust_bass(thumb_status):
    if thumb_status == "thumb_up":
        st.write("Thumb up. Increasing bass.")
        # Implement bass increase logic
    elif thumb_status == "thumb_down":
        st.write("Thumb down. Decreasing bass.")
        # Implement bass decrease logic

# Streamlit UI continued...
st.write("Listening for voice command...")
while True:
    command = recognize_speech()

    if command:
        if "volume" in command:
            st.write("Controlling Volume...")
            control_audio('volume')
        elif "control" in command:
            st.write("Controlling Bass...")
            control_audio('bass')
        elif "exit" in command:
            st.write("Exiting...")
            break
        else:
            st.write("Command not recognized. Try saying 'volume' or 'bass'.")

# Add footer using HTML
st.write("""
    <div style="text-align:center; padding: 20px;">
        <p>Created by B2 team</p>
    </div>
    """, unsafe_allow_html=True)

