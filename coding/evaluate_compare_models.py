import os  
import numpy as np  # helps handle lists of numbers and arrays
import tensorflow as tf  # loads TensorFlow so we can run the models
import matplotlib.pyplot as plt  # used for drawing graphs and plots
import cv2  # used for loading and resizing images
from datetime import datetime  # lets us create timestamps for saved files
from sklearn.metrics import (  # tools for evaluating model performance
    confusion_matrix,  #creates a table showing correct/incorrect predictions
    roc_curve,  #creates data for plotting the ROC curve
    auc,  #calculates the area under a curve (AUC)
    precision_recall_curve,  #creates data for precision‑recall curves
    classification_report  # generates accuracy, precision, recall, F1 scores
)
import csv  # Lets us write results into CSV files

# ================= CONFIG =================
CONFIG = {
    "base_dir": r"C:\LJ_Diss",  # main project folder
    "train_dir": r"C:\LJ_Diss\coding\images\train2",  #training images
    "val_dir": r"C:\LJ_Diss\coding\images\validation2",  # validation images
    "test_dir": r"C:\LJ_Diss\coding\images\test2",  # test images
    "img_size": (224, 224),  # size all images will be resized to
    "batch_size": 32,  # number of images processed at once
    "model_dir": r"C:\LJ_Diss\outputs\models",  # folder where trained models are saved
    "eval_dir": r"C:\LJ_Diss\coding\evaluation_outputs",  # folder where evaluation results will be saved
    "n_gradcam_samples": 10  # number of Gradcam examples to generate
}

os.makedirs(CONFIG["eval_dir"], exist_ok=True)  # creates the evaluation folder if it doesn't exist
AUTOTUNE = tf.data.AUTOTUNE  # lets TensorFlow speed up data loading automatically
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # creates a timestamp for saved files

# ================= MODEL LOADING =================
def get_latest_model_by_type(model_dir, keyword):  #finds the newest model containing a keyword
    models = [f for f in os.listdir(model_dir)
              if f.endswith(".keras") and keyword in f.lower()]  # filters models by name
    if not models:  # if none found
        raise FileNotFoundError(f"No .keras models with '{keyword}' found")  # stop with error
    models.sort()  # sort alphabetically so newest is last
    return os.path.join(model_dir, models[-1])  # return full path to newest model

mobilenet_path = get_latest_model_by_type(CONFIG["model_dir"], "mobilenet")  # find newest MobileNet model
efficientnet_path = get_latest_model_by_type(CONFIG["model_dir"], "efficientnet")  # find newest EfficientNet model

print("Loading MobileNet model:", mobilenet_path)  # show which MobileNet model is loading
print("Loading EfficientNet model:", efficientnet_path)  # show which EfficientNet model is loading

mobilenet_model = tf.keras.models.load_model(mobilenet_path)  # load MobileNet model
efficientnet_model = tf.keras.models.load_model(efficientnet_path)  # load EfficientNet model

# ================= DATASETS =================
def load_raw_dataset(path, shuffle=True):  # loads images from a folder
    ds = tf.keras.utils.image_dataset_from_directory(
        path,  # folder path
        image_size=CONFIG["img_size"],  # resize images
        batch_size=CONFIG["batch_size"],  # batch size
        label_mode="binary",  # labels are 0 or 1 - binary
        shuffle=shuffle  # shuffle if needed
    )
    return ds, ds.class_names  # returns dataset and class names

train_raw, class_names = load_raw_dataset(CONFIG["train_dir"], shuffle=True)  #load training data
val_raw, _ = load_raw_dataset(CONFIG["val_dir"], shuffle=False)  #load validation data
test_raw, _ = load_raw_dataset(CONFIG["test_dir"], shuffle=False)  #load test data

print("Classes:", class_names)  # print class names (e.g., fake2, real2)

# ================= PREPROCESS =================
def preprocess_mobilenet(x, y):  #preprocessing for MobileNet
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)  # apply MobileNet preprocessing
    return x, tf.cast(y, tf.float32)  # return processed image and label

def preprocess_efficientnet(x, y):  # preprocessing for EfficientNet
    x = tf.keras.applications.efficientnet.preprocess_input(x)  # apply EfficientNet preprocessing
    return x, tf.cast(y, tf.float32)  #return processed image and label

def make_ds_for_model(raw_ds, model_type):  # creates dataset for a specific model
    if model_type == "mobilenet":  # if MobileNet
        return raw_ds.map(preprocess_mobilenet, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    else:  # if EfficientNet
        return raw_ds.map(preprocess_efficientnet, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ================= EVALUATION HELPERS =================
def collect_predictions(model, ds):  # runs model on dataset and collects predictions
    y_true, y_prob = [], []  # lists for true labels and predicted probabilities
    for x, y in ds:  # loop through batches
        preds = model.predict(x, verbose=0)  #model predictions
        y_true.extend(y.numpy().ravel().tolist())  #add true labels
        y_prob.extend(preds.ravel().tolist())  #add predicted probabilities
    y_true = np.array(y_true)  #convert to arrays
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)  #convert probabilities to 0/1 predictions
    return y_true, y_prob, y_pred

def plot_confusion_matrix(cm, classes, title, save_path):  #draws confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))  # creates figure
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  #displays matrix
    ax.figure.colorbar(im, ax=ax)  # add colour bar
    ax.set(
        xticks=np.arange(cm.shape[1]),  # X‑axis 
        yticks=np.arange(cm.shape[0]),  # Y‑axis 
        xticklabels=classes,  # X labels
        yticklabels=classes,  # Y labels
        ylabel='True label',  # Y label
        title=title  
    )
    ax.set_xlabel('Predicted label')  # X label

    thresh = cm.max() / 2.  #threshold for text colour
    for i in range(cm.shape[0]):  #loop rows
        for j in range(cm.shape[1]):  #loop columns
            ax.text(j, i, format(cm[i, j], 'd'),  #write number
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")  #choose text colour
    fig.tight_layout()  # Adjust layout
    plt.savefig(save_path, dpi=200)  #save image
    plt.close(fig)  #close figure

def plot_roc_pr(y_true, y_prob, model_name, save_prefix):  #Draw ROC + PR curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)  # ROC curve data
    roc_auc = auc(fpr, tpr)  # ROC AUC score

    plt.figure()  # New figure
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")  #plot ROC
    plt.plot([0, 1], [0, 1], 'k--')  #diagonal line
    plt.xlabel("False Positive Rate")  # X label
    plt.ylabel("True Positive Rate")  # Y label
    plt.title(f"ROC Curve - {model_name}")  # Title
    plt.legend(loc="lower right")  #Legend
    plt.savefig(save_prefix + "_roc.png", dpi=200)  #Save ROC
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)  #PR curve data
    pr_auc = auc(recall, precision)  #PR AUC score

    plt.figure()  #New figure
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")  #plot PR
    plt.xlabel("Recall")  #X label
    plt.ylabel("Precision")  #Y label
    plt.title(f"Precision-Recall Curve - {model_name}")  # itle
    plt.legend(loc="lower left")  # Legend
    plt.savefig(save_prefix + "_pr.png", dpi=200)  #Save PR
    plt.close()

    return roc_auc, pr_auc  #return both AUC scores

# ================= FIXED GRAD-CAM =================
def gradcam_for_model(model, model_type, img_array):  # creates Grad‑CAM heatmap
    feature_layer = model.get_layer("features")  # get feature layer

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,  #model input
        outputs=[feature_layer.output, model.output]  # Outputs: features + prediction
    )

    with tf.GradientTape() as tape:  #track gradients
        features, preds = grad_model(img_array)  #forward pass
        loss = preds[:, 0]  #focus on class score

    grads = tape.gradient(loss, features)  #compute gradients
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # average gradients

    cam = tf.reduce_sum(weights * features[0], axis=-1)  # Weighted sum
    cam = tf.maximum(cam, 0)  # remove negatives
    cam /= tf.reduce_max(cam) + 1e-8  # normalise

    return cam.numpy(), preds.numpy()[0, 0]  # return heatmap + probability

def preprocess_single_image(path, model_type):  # loads and prepares one image for Grad‑CAM
    img = tf.keras.preprocessing.image.load_img(path, target_size=CONFIG["img_size"])  
    # Loads the image from disk and resizes it to the model’s required size

    arr = tf.keras.preprocessing.image.img_to_array(img)  
    # Converts the image into a numerical array so the model can read it

    arr = np.expand_dims(arr, axis=0)  
    # Adds a “batch dimension” because models expect input in batches, even if it's just one image

    if model_type == "mobilenet":  
        # if the image is for MobileNet, apply MobileNet’s preprocessing
        arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    else:
        # Otherwise apply EfficientNet’s preprocessing
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return arr  # Returns the prepared image ready for prediction or Grad‑CAM


def save_side_by_side_models(img_path, cam_m, cam_e, save_path):  
    # saves the original image next to both Grad‑CAM heatmaps
    img = cv2.imread(img_path)  
    #loads the original image from disk

    img = cv2.resize(img, CONFIG["img_size"])  
    # resizes the original image to match the model’s input size

    cam_m_hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam_m, CONFIG["img_size"])), cv2.COLORMAP_JET)  
    # converts MobileNet’s Grad‑CAM map into a colourful heatmap

    cam_e_hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam_e, CONFIG["img_size"])), cv2.COLORMAP_JET)  
    # converts EfficientNet’s Grad‑CAM map into a colourful heatmap

    combined = np.hstack([img, cam_m_hm, cam_e_hm])  
    # places the original image, MobileNet heatmap, and EfficientNet heatmap side‑by‑side

    cv2.imwrite(save_path, combined)  
    # saves the combined comparison image to disk


# ================= FORENSICS CSV =================
def write_forensics_report(rows, save_path):  #writes all prediction results into a CSV file
    header = [
        "image_path",          # Where the image came from
        "true_label",          # The correct class (0 or 1)
        "mobilenet_prob",      # MobileNet’s probability score
        "mobilenet_pred",      # MobileNet’s final prediction (0 or 1)
        "efficientnet_prob",   # EfficientNet’s probability score
        "efficientnet_pred"    # EfficientNet’s final prediction (0 or 1)
    ]

    with open(save_path, "w", newline="") as f:  
        # Opens the CSV file for writing
        writer = csv.writer(f)  
        # Creates a CSV writer object

        writer.writerow(header)  
        # writes the column names at the top of the file

        writer.writerows(rows)  
        # writes one row per image with all prediction details


# ================= MAIN =================
if __name__ == "__main__":  # ensures this code only runs when the script is executed directly

    train_m = make_ds_for_model(train_raw, "mobilenet")  
    # creates a MobileNet‑ready training dataset

    val_m = make_ds_for_model(val_raw, "mobilenet")  
    # creates a MobileNet‑ready validation dataset

    test_m = make_ds_for_model(test_raw, "mobilenet")  
    # creates a MobileNet‑ready test dataset

    train_e = make_ds_for_model(train_raw, "efficientnet")  
    # creates an EfficientNet‑ready training dataset

    val_e = make_ds_for_model(val_raw, "efficientnet")  
    # creates an EfficientNet‑ready validation dataset

    test_e = make_ds_for_model(test_raw, "efficientnet")  
    # creates an EfficientNet‑ready test dataset

    print("\nEvaluating MobileNet on test set...")  
    

    y_true_m, y_prob_m, y_pred_m = collect_predictions(mobilenet_model, test_m)  
    # runs MobileNet on the test set and collects true labels, probabilities, and predictions

    print("Evaluating EfficientNet on test set...")  
   

    y_true_e, y_prob_e, y_pred_e = collect_predictions(efficientnet_model, test_e)  
    #runs EfficientNet on the test set and collects true labels, probabilities, and predictions

    assert np.array_equal(y_true_m, y_true_e), "True labels mismatch"  
    # ensures both models saw the same labels in the same order

    y_true = y_true_m  
    # stores the shared true labels

    cm_m = confusion_matrix(y_true, y_pred_m)  
    # creates MobileNet’s confusion matrix

    cm_e = confusion_matrix(y_true, y_pred_e)  
    # creates EfficientNet’s confusion matrix

    plot_confusion_matrix(cm_m, class_names,
                          "Confusion Matrix - MobileNet",
                          os.path.join(CONFIG["eval_dir"], "mobilenet_confusion.png"))  
    #saves MobileNet’s confusion matrix as an image

    plot_confusion_matrix(cm_e, class_names,
                          "Confusion Matrix - EfficientNet",
                          os.path.join(CONFIG["eval_dir"], "efficientnet_confusion.png"))  
    # saves EfficientNet’s confusion matrix as an image

    roc_m, pr_m = plot_roc_pr(y_true, y_prob_m,
                              "MobileNetV3Small",
                              os.path.join(CONFIG["eval_dir"], "mobilenet"))  
    # creates and saves MobileNet’s ROC and PR curves

    roc_e, pr_e = plot_roc_pr(y_true, y_prob_e,
                              "EfficientNetB0",
                              os.path.join(CONFIG["eval_dir"], "efficientnet"))  
    # creates and saves EfficientNet’s ROC and PR curves

    report_m = classification_report(y_true, y_pred_m, target_names=class_names)  
    # generates MobileNet’s classification report

    report_e = classification_report(y_true, y_pred_e, target_names=class_names)  
    # generates EfficientNet’s classification report

    with open(os.path.join(CONFIG["eval_dir"], "mobilenet_classification_report.txt"), "w") as f:
        f.write(report_m)  
        # saves MobileNet’s report to a text file

    with open(os.path.join(CONFIG["eval_dir"], "efficientnet_classification_report.txt"), "w") as f:
        f.write(report_e)  
        # saves EfficientNet’s report to a text file

    print("\nBuilding forensics report...")  
    

    rows = []  # will store all CSV rows
    image_paths = []  # stores paths to test images
    labels = []  # stores true labels for each image

    for class_idx, cls in enumerate(class_names):  
        # loops through each class (fake2, real2)

        cls_dir = os.path.join(CONFIG["test_dir"], cls + "2") if cls + "2" in os.listdir(CONFIG["test_dir"]) else os.path.join(CONFIG["test_dir"], cls)
        # handles folder naming differences (fake vs fake2)

        if not os.path.isdir(cls_dir):  
            continue  #skip if folder doesn’t exist

        for fname in sorted(os.listdir(cls_dir)):  
            # loops through all images in the class folder

            if fname.lower().endswith((".jpg", ".jpeg", ".png")):  
                #only process image files

                image_paths.append(os.path.join(cls_dir, fname))  
                # save full path to image

                labels.append(class_idx)  
                # save the true label (0 or 1)

    labels = np.array(labels)  
    # convert labels to a NumPy array

    if len(labels) == len(y_true):  
        # ig the number of images matches the number of predictions

        for i, img_path in enumerate(image_paths):  
            rows.append([
                img_path,            # Path to image
                int(labels[i]),      # True label
                float(y_prob_m[i]),  # MobileNet probability
                int(y_pred_m[i]),    # MobileNet prediction
                float(y_prob_e[i]),  # EfficientNet probability
                int(y_pred_e[i])     # EfficientNet prediction
            ])
    else:
        #fallback if something doesn’t match up

        for i in range(len(y_true)):
            rows.append([
                f"index_{i}",         # Use index instead of path
                int(y_true[i]),       # True label
                float(y_prob_m[i]),   # MobileNet probability
                int(y_pred_m[i]),     # MobileNet prediction
                float(y_prob_e[i]),   # EfficientNet probability
                int(y_pred_e[i])      # EfficientNet prediction
            ])

    csv_path = os.path.join(CONFIG["eval_dir"], f"forensics_report_{timestamp}.csv")  
    #path where the CSV will be saved

    write_forensics_report(rows, csv_path)  
    #writes all results to the CSV file

    print("\nGenerating Grad-CAM comparison samples...")  
   

    gradcam_dir = os.path.join(CONFIG["eval_dir"], "gradcam_comparison")  
    #folder where Grad‑CAM images will be saved

    os.makedirs(gradcam_dir, exist_ok=True)  
    #create folder if it doesn’t exist

    sample_count = 0  # Counter for Grad‑CAM samples

    for img_path, true_label in zip(image_paths, labels):  
        #loop through each test image

        if sample_count >= CONFIG["n_gradcam_samples"]:  
            break  # stop once we reach the requested number of samples

        arr_m = preprocess_single_image(img_path, "mobilenet")  
        #preprocess image for MobileNet

        arr_e = preprocess_single_image(img_path, "efficientnet")  
        #preprocess image for EfficientNet

        cam_m, prob_m = gradcam_for_model(mobilenet_model, "mobilenet", arr_m)  
        # generate MobileNet Grad‑CAM

        cam_e, prob_e = gradcam_for_model(efficientnet_model, "efficientnet", arr_e)  
        # generate EfficientNet Grad‑CAM

        base_name = os.path.splitext(os.path.basename(img_path))[0]  
        # extract filename without extension

        save_side_by_side_models(
            img_path,
            cam_m,
            cam_e,
            os.path.join(gradcam_dir, f"{base_name}_mn_vs_en.jpg")
        )  
        # save original + both heatmaps side‑by‑side

        sample_count += 1  
        # increase sample counter

    print("\n=== EVALUATION SUMMARY ===")  
    # final summary header

    print("MobileNet ROC AUC:", roc_m)  
    # MobileNet ROC score

    print("MobileNet PR AUC :", pr_m)  
    # MobileNet PR score

    print("EfficientNet ROC AUC:", roc_e)  
    # EfficientNet ROC score

    print("EfficientNet PR AUC :", pr_e)  
    # EfficientNet PR score

    print("\nAll outputs saved in:")  
    # Message showing where results are stored

    print(CONFIG["eval_dir"])  
    # Prints the evaluation folder path
