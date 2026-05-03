'''MobileNetV3Small - train.py'''
#all the imports from library
import os                     
from datetime import datetime 
import tensorflow as tf   # main deep learning library used for training. - TensorFlow
import numpy as np   # for deadling with arrays and numerical data
import cv2    # for processing each image when saving Gradcam + Gradcam Plus outputs

# ================= CONFIGURATION ==========================================================================
# this section stores all important settings and folder locations in one place
CONFIG = {
    "train_dir": r"C:\LJ_Diss\coding\images\train2",   # path to training images - train directory
    "val_dir": r"C:\LJ_Diss\coding\images\validation2",   # path to validatw images
    "test_dir": r"C:\LJ_Diss\coding\images\test2",   # path to test images
    "img_size": (224, 224),   # images resized - for MobileNetV3Small
    "batch_size": 16,   # number of images processed at once - each batch = 16images
    "epochs_head": 10,   # first training stage length
    "epochs_finetune": 40,   #second training stage length   50 epoch
    "model_dir": r"C:\LJ_Diss\outputs\models",   # trained models saved in this path
    "gradcam_dir": r"C:\LJ_Diss\outputs\gradcam_outputs\mobilenet"  # Gradcam images saved in this path
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)   # creates the model folder if it does not exist
os.makedirs(CONFIG["gradcam_dir"], exist_ok=True)   # creates the Gradcam output folder if it does not exist

AUTOTUNE = tf.data.AUTOTUNE # gets TensorFlow to optimise data loading automatically
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # creates timestamp (at the moment timestamp - format:year, month, day_hour, mins, secs)for the files saved

# ================= DATA LOADING ============================================================
def load_dataset(path, augment=False):
# Loads images from the specified folder and prepares them for training
# TensorFlow automatically assigns labels based on the folder names

    ds = tf.keras.utils.image_dataset_from_directory(
        path,   # folder containing the images
        image_size=CONFIG["img_size"],   #resize images to 224×224
        batch_size=CONFIG["batch_size"],   #number of images per batch
        label_mode="binary",   #labels =0 or 1 - binary
        shuffle=True    #images in shuffled up order
    )

    class_names = ds.class_names   #stores the folder names as class labels

    def preprocess(x, y):
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)  # applies MobileNetV3 preprocessing
        return x, tf.cast(y, tf.float32)   # ensures labels are float values eg;32

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE) #applies preprocessing to all batches.

    if augment:
        aug = tf.keras.Sequential([   
        # a set of random image transformations applied during training
        # transformations (flip, rotation, zoom, contrast changes) create slightly altered
        # versions of the original images to help the model learn to recognise patterns
        # all under different conditions and prevents it from memorising the training data

            tf.keras.layers.RandomFlip("horizontal"),  # This (at random) flips the image left‑to‑right- teaches the model that the direction a face is facing should not affect whether it is real or fake
            tf.keras.layers.RandomRotation(0.1),  # This rotates the image slightly in either direction - helps the model handle faces that are tilted or photographed at a slight angle
            tf.keras.layers.RandomZoom(0.1),   # This zooms in or out by a small amount - prepares the model for faces that appear at different distances from the camera
            tf.keras.layers.RandomContrast(0.1),    # This changes the contrast of the image a little - helps the model deal with different lighting conditions, such as bright or dim environments.
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),        
                    num_parallel_calls=AUTOTUNE)
        # applies the augmentation pipeline to each batch of images
        # Only the images are changed-the labels stay the same
        # each time an image passes through this function, it may be flipped, rotated, zoomed, or contrast‑adjusted. This creates new variations of the same image, helping the model learn to generalise rather than
        # memorise the training data.

    return ds.prefetch(AUTOTUNE), class_names                        # Prefetch improves training speed.
    # prefetching prepares the next batch of images while the model is training on the current one. This keeps the GPU or CPU busy and
    # speeds up training by reducing waiting time.


# ================= MODEL DEFINITION =================
def build_model():
#Builds a MobileNetV3Small model with a custom classifier and a named layer for Gradcam
    
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), # input image shape (resized)
        include_top=False,  # excludes the default classifier
        weights="imagenet"  # loads pretrained ImageNet weights
    )
    base.trainable = False  # freezes the base model for the first training stage 

    inputs = tf.keras.Input(shape=(224, 224, 3))  # defines the model input
    x = base(inputs, training=False)   # passes input through the pretrained base

    x = tf.keras.layers.Layer(name="features")(x) # adds a named layer for gradcam access

    x = tf.keras.layers.GlobalAveragePooling2D()(x) # converts feature maps into a single vector
    x = tf.keras.layers.BatchNormalization()(x)  # normalises values to stabilise training

    x = tf.keras.layers.Dense(256, activation="relu")(x) # fully connected layer for learning image patterns
    x = tf.keras.layers.Dropout(0.5)(x)  #reduces overfitting by randomly dropping units

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x) #final output: probability of real/fake.

    model = tf.keras.Model(inputs, outputs) # combines everything into a full model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),   # learning rate for the first stage
        loss="binary_crossentropy",  # standard loss for binary classification
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]  #Tells the model to report two evaluation measures during training:
    # 1. accuracy- the percentage of predictions the model gets correct
    # 2. AUC- a measure of how well the model separates the two classes (real vs fake) across all possible decision thresholds
    # 3. this helps monitor how well the model is learning over time
    )

    return model, base

# ================= GRAD-CAM UTILITIES =================
def preprocess_image(img_path):
#loads and prepares a single image for Gradcam

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224)) #loads and resizes image
    arr = tf.keras.preprocessing.image.img_to_array(img)   #converts to array
    arr = np.expand_dims(arr, axis=0) #adds batch dimension
    return tf.keras.applications.mobilenet_v3.preprocess_input(arr) #prepares the image in the format MobileNetV3 was trained on

def gradcam(model, img_array):
#generates a Gradcam heatmap showing which areas influenced the model's decision

    feature_layer = model.get_layer("features")   # retrieves the named feature layer

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,  #takes the original model input
        outputs=[feature_layer.output, model.output] # Outputs both features and prediction
    )

    with tf.GradientTape() as tape:  # Records operations for gradient calculation
        features, preds = grad_model(img_array)   # gets feature maps and prediction
        loss = preds[:, 0]   # uses the prediction score as the target

    grads = tape.gradient(loss, features)  #gets how much each feature map contributed to the model’s prediction

    weights = tf.reduce_mean(grads, axis=(0, 1, 2)) # averages gradients to get importance weights

    cam = tf.reduce_sum(weights * features[0], axis=-1) # puts together weights with feature maps
    cam = tf.maximum(cam, 0)  #Keeps only the positive values
    cam /= tf.reduce_max(cam) + 1e-8 #normalises heatmap to 0–1.

    return cam.numpy(), preds.numpy()[0, 0] #returns heatmap and prediction score

def gradcam_pp(model, img_array):
# creates a Gradcam++ heatmap, which usually  produces sharper visualisations.
    
    feature_layer = model.get_layer("features")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape: # multiple gradient calculations
        features, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, features)   #  first‑order gradients: shows how each feature map affects the prediction
    grads2 = grads ** 2  # second‑order gradients: squares the gradients for Gradcam++ -Looks at how the first gradient itself changes
#highlights areas where the model’s influence is stronger or more certain
    grads3 = grads ** 3  #third‑order gradients: cubes the gradients for Gradcam++ -Goes one level deeper again
#exaggerates strong signals even more and helps Gradcam++ focus on the most important regions
    sum_features = tf.reduce_sum(features, axis=(1, 2)) # Sums feature maps

    alpha = grads2 / (2 * grads2 + grads3 * sum_features[:, None, None, :] + 1e-8) 
    # calculates a weighting factor used in Gradcam++
    # combines the 2nd‑ and 3rd‑order gradients with the feature map values
    #  decides how important each channel is for the prediction
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))
    # turns those weighting factors into a single importance value per channel by summing across the height and width of the feature maps

    cam = tf.reduce_sum(weights[:, None, None, :] * features, axis=-1)
    # creates the heatmap by combining the channel weights with the feature maps- shows which areas of the image influenced the prediction
    cam = tf.maximum(cam, 0) # removes negative values so only positive, meaningful regions remain
    cam /= tf.reduce_max(cam) + 1e-8 ## normalises the heatmap to a 0–1 range so it can be displayed properly

    return cam[0].numpy(), preds.numpy()[0, 0] #returns the final heatmap and the model’s prediction score.

def save_overlay(img_path, heatmap, save_path):
 #saves a blended image showing the original picture with the heatmap on top

    img = cv2.imread(img_path)  # loads original image
    img = cv2.resize(img, (224, 224)) #resizes to match heatmap

    heatmap = cv2.resize(heatmap, (224, 224))  #resizes heatmap.
    heatmap = np.uint8(255 * heatmap)  # converts to 0–255 scale
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #applies colour map

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0) # blends the image and heatmap
    cv2.imwrite(save_path, overlay) # Saves that image

def save_side_by_side(img_path, cam, campp, save_path):
# saves image showing three panels: the original image,the Gradcam heatmap, and the Gradcam++ heatmap.  
    img = cv2.imread(img_path) #loads the original image from the file path
    img = cv2.resize(img, (224, 224)) # resizes the original image so it matches the heatmap size

    cam_hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (224, 224))), cv2.COLORMAP_JET)
    # converts the Gradcam heatmap into a coloured image (JET colour map) after resizing it and scaling values to 0–255
    campp_hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(campp, (224, 224))), cv2.COLORMAP_JET)
    # does the same as above, but for the Gradcam++ heatmap
    combined = np.hstack([img, cam_hm, campp_hm])  # Places all three images side by side
    # places the three images horizontally next to each other.
    cv2.imwrite(save_path, combined) #    # saves the final side‑by‑side comparison image to the given file path


# ================= TRAINING AND GRAD-CAM EXECUTION =========================================================
if __name__ == "__main__":

    train_ds, class_names = load_dataset(CONFIG["train_dir"], augment=True) # loads training data
    val_ds, _ = load_dataset(CONFIG["val_dir"], augment=False)   # loads validation data

    print("Classes:", class_names)

    model, base = build_model()  #builds the model and retrieves the base network

    print("\n Phase 1: Training head")
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["epochs_head"]) # trains classifier only

    print("\n Phase 2: Fine-tuning")
    base.trainable = True   # unfreezes the base model for fine‑tuning

    for layer in base.layers[:-60]: #keeps earlier layers frozen to avoid overfitting.
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-6), #lower learning rate for fine‑tuning
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["epochs_finetune"]) #fine‑tunes model.

    model_path = os.path.join(CONFIG["model_dir"], f"mobilenetv3_50ep_{timestamp}.keras")
    model.save(model_path)       #finally saves the trained model
    print("\n Saved:", model_path)

    print("\nRunning Gradcam on test images...")

    for cls in ["fake2", "real2"]:  #loops over both test classes: fake2 and real2
        input_dir = os.path.join(CONFIG["test_dir"], cls) #builds path to the folder containing test images for this class
        output_dir = os.path.join(CONFIG["gradcam_dir"], cls) # builds path where Gradcam outputs will be saved
        os.makedirs(output_dir, exist_ok=True) # creates the output folder if non-existant

        for file in os.listdir(input_dir):  # loops through every file in the class folder
            if not file.lower().endswith((".jpg", ".png", ".jpeg")): # skips files that are not images
                continue

            img_path = os.path.join(input_dir, file) # full path to the current image
            arr = preprocess_image(img_path)  # loads, resizes, and preprocess the image for the model

            cam, _ = gradcam(model, arr)  #creates Gradcam heatmap
            campp, _ = gradcam_pp(model, arr) #creates Gradcam++ heatmap

            name = os.path.splitext(file)[0]  # extracts the filename without extension

            save_overlay(img_path, cam, os.path.join(output_dir, f"{name}_cam.jpg")) # Saves Gradcam overlay on the original image
            save_overlay(img_path, campp, os.path.join(output_dir, f"{name}_campp.jpg")) #saves Gradcam++ overlay on the original image
            save_side_by_side(img_path, cam, campp, os.path.join(output_dir, f"{name}_combined.jpg") ) #saves a 3‑panel comparison image

    print("\nGradcam complete for MobileNetV3Small.")
