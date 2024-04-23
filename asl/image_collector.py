import os
import cv2
from asl.globals import *

def run(label_classes=["NONE",]): 
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    passed_classes = []
    if label_classes[0] == "NONE":
        passed_classes = range(NUM_CLASSES)
    else:
        for label in label_classes:
            passed_classes.append(CLASS_MAP[label])
    
    cap = cv2.VideoCapture(0)
    for i in passed_classes:
        if not os.path.exists(os.path.join(TEMP_DIR, str(i))):
            os.makedirs(os.path.join(TEMP_DIR, str(i)))
        
        fname = os.path.join(TEMP_DIR, str(i), META)
        f = open(fname, 'w')
        f.write('start_counter={}'.format(DATASET_SIZE))
        f.close()

        print('Collecting data for class {}'.format(i))

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (50, 50), (300, 350), (255, 0, 0), 2)
            cv2.putText(frame, 'Ready? Press Enter', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            if ret:
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) == ENTER_KEY:
                    break
            else:
                print('Error reading from video frame')
                break

        counter = 0
        while counter < DATASET_SIZE:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if ret:
                cv2.putText(frame, LABEL_MAP[i], (55, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (50, 50), (300, 350), (255, 0, 0), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(TEMP_DIR, str(i), '{}.jpg'.format(counter)), frame)
                counter += 1
            else:
                print('Error reading from video frame')
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()