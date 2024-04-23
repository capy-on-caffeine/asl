import sys
import asl.image_collector as collector
import asl.make_dataset as make
import asl.classifier as classifier
import asl.instance_runner as runner
import asl.cleanup as cleanup
from asl.globals import *


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
        
        collector.run(label_classes=labels)
        make.run(mode="evaluation")
        classifier.run(mode="evaluation")
        runner.run(mode="evaluation", label_classes=labels)
        
    elif args[0] == "predict":
        runner.run(mode="normal")
        
    elif args[0] == "clean":
        cleanup.run()
        
    else:
        print("Error: Invalid command")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
