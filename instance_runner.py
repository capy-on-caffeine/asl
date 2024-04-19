import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import make_dataset
import classifier
import shutil


def run(mode='evaluation'):
    ENTER_KEY = 13
    ESC_KEY = 27
    
    number_of_classes = 3
    
    deploy_data = False
    
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    folder = ''    
    if mode == 'evaluation':
        folder = 'temp_files'
    else:
        folder = 'deploy'
        
    model = pickle.load(open('{}/model.p'.format(folder), 'rb'))['model']

    labels_dict = {0: 'A', 1: 'B', 2: 'L'}

    while True:
        data_aux = []
        x_ = []
        y_ = []
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        H, W, _ = frame.shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
            
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            
            try:            
                pred = model.predict([np.asarray(data_aux)])
                label_predicted = labels_dict[int(pred[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, label_predicted, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            except ValueError:
                pass
            
        if ret:
            cv2.putText(frame, 'Enter - accept', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Esc - reject', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (50, 50), (300, 350), (255, 0, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ENTER_KEY:
                deploy_data = True
                break
            if cv2.waitKey(1) == ESC_KEY:
                deploy_data = False
                break
        else:
            print('Error reading from video frame')
            break
        
        
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    
    if mode == 'evaluation':    
        if deploy_data:
            # append temp_files to data
            for i in range(number_of_classes):
                
                f = open('temp/{}/meta.txt'.format(i), 'r')
                file_num = f.readline().split('=')[1]
                f.close()
                
                files = os.listdir('./temp')
                for file in files:
                    if file.endswith('.jpg'): ## remove this as this does not copy the metadata
                        source_file = os.path.join('temp', i, file)
                        if not os.path.exists(os.path.join('./data', i)):
                            os.makedirs(os.path.join('./data', i))
                        destination_file = os.path.join('./data', i, file_num)
                        shutil.copyfile(source_file, destination_file)
                        file_num += 1
            # make new dataset
            make_dataset.run(mode='normal')
            # train model
            classifier.run(mode='normal')
            
            shutil.rmtree('./temp')
            shutil.rmtree('./temp_files')
        else:
            # delete temp_files
            shutil.rmtree('./temp')
            shutil.rmtree('./temp_files')

if __name__ == "__main__":
    run(mode='normal')