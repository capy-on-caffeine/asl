import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run(mode='evaluation'):
    folder = ''
    if mode == 'evaluation':
        folder = 'temp_files'
    else:
        folder = 'deploy'
        
    data_dict = pickle.load(open('{}/data.pickle'.format(folder), 'rb'))
    
    try:
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])
    except ValueError:
        print("Data unfit for model training. Exiting pipeline...")
        # CLEANUP FUNCTION HERE ############################################################
        exit()

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy: ', accuracy_score(y_pred, y_test))

    f = open('{}/model.p'.format(folder), 'wb')
    pickle.dump({'model': model}, f)
    f.close()

if __name__ == "__main__":
    run(mode='normal')
    # run(mode='evaluation')