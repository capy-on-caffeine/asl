import os
import cv2

def run():
    DATA_DIR = './temp' # always dumps to temp
    ENTER_KEY = 13
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    number_of_classes = 3
    dataset_size = 200

    cap = cv2.VideoCapture(0)
    for i in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(i))):
            os.makedirs(os.path.join(DATA_DIR, str(i)))
        
        fname = DATA_DIR + "/" + str(i) + "/meta.txt"
        f = open(fname, 'w')
        f.write('start_counter={}'.format(dataset_size))
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
        while counter < dataset_size:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if ret:
                cv2.rectangle(frame, (50, 50), (300, 350), (255, 0, 0), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(counter)), frame)
                counter += 1
            else:
                print('Error reading from video frame')
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()