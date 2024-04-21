# American Sign Language Detector

The model is provided, the data isnt.

# Installation

1. Clone this repo:

   ```
   git clone https://github.com/capy-on-caffeine/asl.git
   ```

2. (Optional) Create a virtual environment:
   ```
   python -m venv env
   env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the program:

   To collect images, run (from root of the project):  
   ```
   asl collect
   ```

   To start using the application right away, run:
   ```
   asl predict
   ```

   Optionally, add the asl.bat file to PATH variable to access this from anywhere ✨