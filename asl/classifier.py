import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import asl.cleanup as cleanup
from asl.globals import *


def run(mode="evaluation"):
    DUMP_DIR = ""
    if mode == "evaluation":
        DUMP_DIR = TEMP_FILES_DIR
    else:
        DUMP_DIR = DEPLOY_DIR
    
    fname = os.path.join(DUMP_DIR, DATASET)
    data_dict = pickle.load(open(fname, "rb"))

    try:
        data = np.asarray(data_dict["data"])
        labels = np.asarray(data_dict["labels"])
    except ValueError:
        print("Data unfit for model training. Deleting temp files...")
        cleanup.run()
        print("Exiting pipeline...")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, shuffle=True, stratify=labels
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_pred, y_test) * 100, "%")
    
    fname = os.path.join(DUMP_DIR, MODEL)
    f = open(fname, "wb")
    pickle.dump({"model": model}, f)
    f.close()


if __name__ == "__main__":
    run(mode="normal")
    # run(mode='evaluation')
