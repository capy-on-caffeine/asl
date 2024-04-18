import image_collector
import make_dataset
import classifier
import instance_runner

def main():
    print("Welcome to MyApp!")
    print("What do you want to do?")
    print("1. Collect images")
    print("2. Make dataset")
    print("3. Run classifier")
    print("4. Run instance runner")
    print("Enter anything else to exit...")
    
    while True:
        choice = input("Enter your choice: ")

        if choice == "1":
            image_collector.run()
        elif choice == "2":
            make_dataset.run()
        elif choice == "3":
            classifier.run()
        elif choice == "4":
            instance_runner.run()
        else:
            break

if __name__ == "__main__":
    main()
