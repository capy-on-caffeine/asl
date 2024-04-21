import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import make_dataset
import classifier
import shutil
import cleanup
from globals import *

def run(mode="evaluation", label_classes=["NONE",]):
    deploy_data = False

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    DUMP_DIR = ""
    if mode == "evaluation":
        DUMP_DIR = TEMP_FILES_DIR
    else:
        DUMP_DIR = DEPLOY_DIR
    
    fname = os.path.join(DUMP_DIR, MODEL)
    model = pickle.load(open(fname, "rb"))["model"]

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * w) - 10
            y1 = int(min(y_) * h) - 10

            x2 = int(max(x_) * w) - 10
            y2 = int(max(y_) * h) - 10

            try:
                pred = model.predict([np.asarray(data_aux)])
                label_predicted = LABEL_MAP[int(pred[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    label_predicted,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            except ValueError:
                pass

        if ret:
            cv2.putText(
                frame,
                "Enter - accept",
                (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Esc - reject",
                (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (50, 50), (300, 350), (255, 0, 0), 2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ENTER_KEY:
                deploy_data = True
                break
            if cv2.waitKey(1) == ESC_KEY:
                deploy_data = False
                break
        else:
            print("Error reading from video frame")
            break

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if mode == "evaluation":
        if deploy_data:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            
            passed_classes = []
            if label_classes[0] == "NONE":
                passed_classes = range(NUM_CLASSES)
            else:
                for label in label_classes:
                    passed_classes.append(CLASS_MAP[label])

            for i in passed_classes:
                file_num = 0
                try:
                    fname = os.path.join(DATA_DIR, str(i), META)
                    f = open(fname, "r")
                    file_num = int(f.readline().split("=")[1])
                    f.close()
                except FileNotFoundError:
                    os.makedirs(os.path.join(DATA_DIR, str(i)))
                    fname = os.path.join(DATA_DIR, str(i), META)
                    f = open(fname, "w")
                    f.write("start_counter={}".format(file_num))
                    # file_num will remain zero in this case
                    f.close()

                files = os.listdir(os.path.join(TEMP_DIR, str(i)))
                for file in files:
                    if file.endswith(
                        ".jpg"
                    ):
                        source_file = os.path.join(TEMP_DIR, str(i), file)
                        if not os.path.exists(os.path.join(DATA_DIR, str(i))):
                            os.makedirs(os.path.join(DATA_DIR, str(i)))
                        destination_file = os.path.join(
                            "data", str(i), "{}.jpg".format(str(file_num))
                        )
                        shutil.copyfile(source_file, destination_file)
                        file_num += 1

                fname = os.path.join(DATA_DIR, str(i), META)
                f = open(fname, "w")
                f.write("start_counter={}".format(file_num))
                f.close()

            make_dataset.run(mode="normal")
            classifier.run(mode="normal")

        cleanup.run()

if __name__ == "__main__":
    run(mode="normal")
    # run(mode="evaluation")
