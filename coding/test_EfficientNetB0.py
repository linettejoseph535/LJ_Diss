'test3.py'
'''EfficientNetB0'''
import os  
import glob  #search for files that match a pattern (like *.keras)
import numpy as np  # Used for handling lists of numbers and arrays
import tensorflow as tf  # Loads TensorFlow, which runs the model and preprocessing
import matplotlib.pyplot as plt  # creates plots and graphs
import seaborn as sns  # Makes nicer‑looking heatmaps and charts
from sklearn.metrics import (  # Imports tools for evaluating model performance
    classification_report,  # creates a summary of accuracy for each class
    confusion_matrix, # shows how many images were correctly/incorrectly classified
    roc_auc_score,  # calculates AUC, a measure of prediction quality
    roc_curve  # creates data for plotting the ROC curve
)


# ================= CONFIG =================
CONFIG = {
    "test_dir": r"C:\LJ_Diss\coding\images\test2",  # folder containing test images
    "batch_size": 16,  # number of images processed at once
    "img_size": (224, 224),  #size each image is resized to
    "model_dir": r"C:\LJ_Diss\outputs\models",  #where trained models are stored
    "plots_dir": r"C:\LJ_Diss\outputs\plots\efficientnet",  # graphs will be saved here
}


os.makedirs(CONFIG["plots_dir"], exist_ok=True)  #creates the plots folder if it doesn't already exist
AUTOTUNE = tf.data.AUTOTUNE  # Lets TensorFlow automatically speed up data loading


# ================= LOAD LATEST EFFICIENTNET MODEL =================  # Section header for finding the newest saved EfficientNet model
def get_latest_efficientnet(model_dir):   #this function - looks for the most recently saved EfficientNet model file
    models = glob.glob(os.path.join(model_dir, "*efficientnet*.keras"))  # # searches the folder for files containing "efficientnet" in the name
    if not models:  #checks if no matching model files were found
        raise ValueError("No EfficientNet models found")  #stops the program with an error message
    return max(models, key=os.path.getctime)  # Returns the newest model based on creation time


# ================= DATA =================  # Section header for loading the test dataset
def load_dataset(path):  # function- loads images from a folder

    ds = tf.keras.utils.image_dataset_from_directory(  #loads images and labels from the folder
        path,  #folder containing the test images
        image_size=CONFIG["img_size"],  # every image resized to 224×224
        batch_size=CONFIG["batch_size"], # loads images in batch of 16
        label_mode="binary",  #labels each image as 0 or 1 - binary
        shuffle=False  # keeps the order the same so predictions match the correct labels
    )


    class_names = ds.class_names  #stores names of the two classes (e.g., fake2, real2)

    def preprocess(x, y):    # function to prepare each image
        x = tf.keras.applications.efficientnet.preprocess_input(x)  #applies EfficientNet’s required preprocessing
        return x, y  # returns the processed image and its label


    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)  # applies preprocessing to every batch efficiently
    ds = ds.prefetch(AUTOTUNE)  # loads data in advance to speed up testing

    return ds, class_names  # returns the dataset and the class names

# ================= CONFUSION MATRIX =================  
def plot_cm(y_true, y_pred, classes, save_path):  #function - draw and save a confusion matrix
    cm = confusion_matrix(y_true, y_pred)  # creates a table showing correct and incorrect predictions

    plt.figure(figsize=(6, 5))  # sets the size of the plot window
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",  # draws the confusion matrix as a coloured grid
                xticklabels=classes, yticklabels=classes)  # labels the axis with class names
    plt.xlabel("Predicted")  # label for the horizontal axis
    plt.ylabel("True")  # label for the vertical axis
    plt.title("EfficientNetB0 Confusion Matrix")  #title of the plot
    plt.savefig(save_path)  #saves the plot as an image file
    plt.close()  # closes the plot so it doesn’t stay open in memory


# ================= MAIN ================= 
if __name__ == "__main__":  # ensures this code only runs when the file is executed directly


    print("Loading dataset...")  # tells user that the test images are being loaded
    test_ds, class_names = load_dataset(CONFIG["test_dir"])  # loads the test dataset and class names

    model_path = get_latest_efficientnet(CONFIG["model_dir"])  # finds the newest EfficientNet model file
    print("Loading model:", model_path)  # prints the path of the model being loaded
    model = tf.keras.models.load_model(model_path)  # loads the trained EfficientNetB0 model

    y_true, y_probs = [], []  #creates two empty lists: one for true labels, one for predicted probabilities

    print("Predicting...")  # tells user that predictions are starting
    for x, y in test_ds:  # loops through each batch of test images
        preds = model.predict(x, verbose=0)  # gets the model’s prediction for the batch
        y_probs.extend(preds.flatten())  # adds the predicted probabilities to the list
        y_true.extend(y.numpy())  # adds the true labels to the list


    y_true = np.array(y_true).astype(int)  # converts true labels into a clean integer array
    y_probs = np.array(y_probs)  # converts predicted probabilities into a numpy array


    # ================= STATS =================  
    print("\nPrediction stats:")  # prints a label before showing the numbers
    print("Min :", y_probs.min())  # shows the lowest prediction score the model produced
    print("Max :", y_probs.max())  # shows the highest prediction score the model produced
    print("Mean:", y_probs.mean())  # shows the average prediction score across all test images


    # ================= AUC =================  
    auc = roc_auc_score(y_true, y_probs)  # calculates how well the model separates real vs fake images
    print("\nAUC:", auc)  # Prints the AUC value


    # ================= BEST THRESHOLD =================  
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)  # creates data showing how predictions behave at different cutoffs
    best_threshold = thresholds[np.argmax(tpr - fpr)]  # picks the cutoff that gives the best balance between correct and incorrect predictions
    print("Best threshold:", best_threshold)  # prints the chosen cutoff value

    y_pred = (y_probs > best_threshold).astype(int)  # converts probabilities into final 0/1 predictions using the chosen cutoff

    # ================= REPORT =================
    print("\nClassification Report:\n")  # prints a label before the report
    print(classification_report(y_true, y_pred, target_names=class_names))  # shows accuracy, precision, recall, and F1-score for each class

    # ================= CONFUSION MATRIX ================= 
    cm_path = os.path.join(CONFIG["plots_dir"], "efficientnet_confusion_matrix.png")  # creates the full file path for the confusion matrix image
    plot_cm(y_true, y_pred, class_names, cm_path)  # creates and saves the confusion matrix plot

    print("\nSaved confusion matrix to:", cm_path)  # tells  user where the confusion matrix image was saved
    print("\n EfficientNetB0 testing complete!")  
