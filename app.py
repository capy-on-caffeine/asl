import sys
import image_collector
import make_dataset
import classifier
import instance_runner

def main(args):
    if args[0] == 'collect':
        image_collector.run()
        make_dataset.run(mode='evaluation')
        classifier.run(mode='evaluation')
        instance_runner.run(mode='evaluation')
    elif args[0] == 'predict':
        instance_runner.run(mode='normal')
    else:
        print("Input error")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
