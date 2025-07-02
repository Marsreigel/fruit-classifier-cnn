from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load your trained model (verified path)
model = load_model("C:/Users/padda/Downloads/fruit_cnn_submission/fruit_classifier_cnn_model.h5")

# All 105 class labels (exactly from original training)
class_labels = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans',
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon', 'apple braeburn',
    'apple golden', 'apple granny smith', 'apple pink lady', 'apple red delicious',
    'apple red yellow', 'apricot', 'avocado', 'banana red', 'cactus fruit', 'cantaloupe',
    'cherry', 'clementine', 'dates', 'granadilla', 'grapefruit pink', 'grapefruit white',
    'guava', 'huckleberry', 'kaki', 'kumquats', 'lemon meyer', 'limes', 'lychee', 'mandarine',
    'mango red', 'melon piel de sapo', 'mulberry', 'nectarine', 'orange bitter', 'orange blood',
    'peach', 'peach flat', 'pear abate', 'pear monster', 'pear williams', 'pepino',
    'physalis', 'physalis with husk', 'pineapple mini', 'pitahaya red', 'plum', 'plum 2',
    'plum 3', 'pomelo sweetie', 'quince', 'rambutan', 'raspberry', 'redcurrant', 'salak',
    'strawberry', 'strawberry wedge', 'tamarillo', 'tangelo', 'tomato cherry red',
    'tomato maroon', 'tomato yellow', 'walnut'
]

# Save predictions history
history_file = "predicted_images_log.txt"

# Function to predict only new images
def predict_custom_images(folder_path):
    # Load or create prediction history
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            predicted_files = set(file.read().splitlines())
    else:
        predicted_files = set()

    print(f"\n📂 Predicting new images from: {folder_path}\n")

    new_predictions = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')) and file not in predicted_files:
            file_path = os.path.join(folder_path, file)
            img = load_img(file_path, target_size=(100, 100))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            pred_label = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            print(f"📌 File: {file} | Predicted: {pred_label} ({confidence:.2f}%)")

            # Log this image as predicted
            new_predictions.append(file)

    # Update log file clearly
    if new_predictions:
        with open(history_file, "a") as file:
            file.write("\n".join(new_predictions) + "\n")
    else:
        print("✅ No new images to predict. All images are already processed.")

# ✅ Your clearly verified folder path:
predict_custom_images("C:/Users/padda/Downloads/Project More Fruit Images")
