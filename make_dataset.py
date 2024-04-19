import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle

def run(mode='normal'):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    if mode == 'evaluation':
        DATA_DIR = './temp'
    else:
        DATA_DIR = './data'

    data = []
    labels = []

    for dir in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
            if img_path.endswith('.txt'):
                continue

            data_aux = []
            img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                
                data.append(data_aux)
                labels.append(dir)

    if not os.path.exists('deploy'):
        os.makedirs('deploy')
    
    if not os.path.exists('temp_files'):
        os.makedirs('temp_files')
    
    folder = ''
    if mode == 'evaluation':
        folder = 'temp_files'
    else:
        folder = 'deploy'
        
    f = open('{}/data.pickle'.format(folder), 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

if __name__ == "__main__":
    run(mode='normal')