import mediapipe as mp
import cv2
import os
import pickle
from asl.globals import *


def run(mode="normal"):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    DIR = ""
    if mode == "evaluation":
        DIR = TEMP_DIR
    else:
        DIR = DATA_DIR

    data = []
    labels = []

    for dir in os.listdir(DIR):
        for img_path in os.listdir(os.path.join(DIR, dir)):
            if img_path.endswith(".txt"):
                continue

            data_aux = []
            img = cv2.imread(os.path.join(DIR, dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                        # apply checks to delete image and remove it from dataset if it's not fit for training
                        # maybe a function that renames images to compact them

                data.append(data_aux)
                labels.append(dir)

    if not os.path.exists(DEPLOY_DIR):
        os.makedirs(DEPLOY_DIR)

    if not os.path.exists(TEMP_FILES_DIR):
        os.makedirs(TEMP_FILES_DIR)

    DUMP_DIR = ""
    if mode == "evaluation":
        DUMP_DIR = TEMP_FILES_DIR
    else:
        DUMP_DIR = DEPLOY_DIR
    
    fname = os.path.join(DUMP_DIR, DATASET)
    f = open(fname, "wb")
    pickle.dump({"data": data, "labels": labels}, f)
    f.close()


if __name__ == "__main__":
    run(mode="normal")
    # run(mode='evaluation')
