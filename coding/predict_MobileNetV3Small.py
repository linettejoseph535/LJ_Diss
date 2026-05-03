'''mobilenet'''

import os  
import numpy as np  # used for handling numbers and arrays
import tensorflow as tf  # loads TensorFlow, which runs the model
import cv2  # used for loading and resizing images


# ================= BASE DIRECTORY =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # gets the folder where this script is located

# ================= FIND LATEST MODEL =================
model_dir = r"C:\LJ_Diss\outputs\models"  # folder where trained models are stored

if not os.path.exists(model_dir): # checks if the folder exists
    raise FileNotFoundError(f"Model directory not found: {model_dir}") #stops the script if missing

model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")] # gets all .keras model files

if not model_files: # checks if no model files were found
    raise FileNotFoundError("No .keras model files found in outputs/models") # stops the script

model_files.sort() # sorts the model files alphabetically (newest is last)
latest_model = model_files[-1] # picks the last file; the newest model

model_path = os.path.join(model_dir, latest_model) # creates the full path to the model file
print(f"Loading model: {model_path}") # shows which model is being loaded

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(model_path) # loads the MobileNet model into memory

# ================= CONFIG =================
CONFIG = {
    "img_size": (224, 224),# size each image will be resized to
}

# ================= PREPROCESS =================
def preprocess(img):  # function - prepare an image for MobileNet
    img = cv2.resize(img, CONFIG["img_size"])  # resizes the image to 224×224
    img = img.astype(np.float32)  # converts the image to float values
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)  # applies MobileNetV3 preprocessing
    return np.expand_dims(img, axis=0)  # adds a batch dimension so the model can read it

# ================= PREDICT SINGLE IMAGE =================
def predict(image_path):  #function- predict a single image
    img = cv2.imread(image_path)  # loads the image from the file

    if img is None:  # checks if the image failed to load
        raise ValueError(f"Image not found: {image_path}")  # stops with an error


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts BGR (blue green red) (OpenCV format) to RGB (red green blue)
    x = preprocess(img)  # preprocesses the image for MobileNet


    prob = model.predict(x, verbose=0)[0][0]  # gets the model’s probability score
    label = int(prob > 0.5)  # converts probability into 0 or 1 using a 0.5 cutoff


    return label, float(prob)  # returns the predicted label and probability


# ================= TEST WHOLE TEST SET =================
def evaluate_test_set():  # function - test the entire dataset
    test_dir = os.path.join(BASE_DIR, "images", "test2")  # builds the path to the test folder

    if not os.path.exists(test_dir):  # checks if the test folder exists
        raise FileNotFoundError(f"Test directory not found: {test_dir}")  # stops if missing

    print("\n Running predictions on test set...\n")  # prints a message before starting

    total = 0  #counts total images
    correct = 0  # counts how many predictions were correct


    for label_name in ["fake2", "real2"]:  # loops through both classes
        class_dir = os.path.join(test_dir, label_name)  # Path to each class folder

        for img_name in os.listdir(class_dir):  # loops through each image in the class
            img_path = os.path.join(class_dir, img_name)  # full path to the image
           
            pred_label, prob = predict(img_path)  #predicts the image


            # Ground truth: fake2=0, real2=1  # explains the true label mapping
            true_label = 0 if label_name == "fake2" else 1  #converts folder name to true label

            if pred_label == true_label:  # checks if prediction is correct
                correct += 1  # increases correct count


            total += 1  # increases total count


            print(f"{img_name} → Pred: {pred_label} | True: {true_label} | Prob: {prob:.4f}")  
            # prints the filename, predicted label, true label, and probability

    accuracy = correct / total if total > 0 else 0  # calculates accuracy safely
    print(f"\n Test Accuracy: {accuracy:.4f} ({correct}/{total})")  # prints final accuracy

# ================= MAIN =================
if __name__ == "__main__":  #runs only if this file is executed directly
    evaluate_test_set()  #runs the full test evaluation
