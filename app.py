import sys
import image_collector
import make_dataset
import classifier
import instance_runner
import cleanup
from globals import *


def main(args):
    if args[0] == "collect":
        labels = args[1:]
        labels = [label.upper() for label in labels]
        
        for label in labels:
            if label.upper() not in LABEL_MAP.values():
                print("Error: Invalid data class entered")
                exit()
            if label == "NONE":
                labels = ["NONE",]
                break
        
        image_collector.run(label_classes=labels)
        make_dataset.run(mode="evaluation")
        classifier.run(mode="evaluation")
        instance_runner.run(mode="evaluation", label_classes=labels)
        
    elif args[0] == "predict":
        instance_runner.run(mode="normal")
        
    elif args[0] == "clean":
        cleanup.run()
        
    else:
        print("Error: Invalid command")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
