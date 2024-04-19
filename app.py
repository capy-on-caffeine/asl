import image_collector
import make_dataset
import classifier
import instance_runner

def main():
    print("What do you want to do?")
    print("1. Collect more images")
    print("2. Run instance")
    print("Enter anything else to exit...")
    
    while True:
        choice = input("Enter your choice: ")

        if choice == "1":
            image_collector.run()
            make_dataset.run(mode='evaluation')
            classifier.run(mode='evaluation')
            instance_runner.run(mode='evaluation')
        elif choice == "2":
            instance_runner.run(mode='normal')
        else:
            break

if __name__ == "__main__":
    main()
