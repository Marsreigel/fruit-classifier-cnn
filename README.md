# fruit-classifier-cnn
Classify images of fruits and vegetables into one of 105 classes.
#  Fruit & Vegetable Image Classifier (CNN)

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify 105 different fruits and vegetables. It is trained on the [Fruits 360 Dataset](https://www.kaggle.com/moltean/fruits), and supports both batch and single-image prediction.

---

##  Dataset Overview

- **Total Images**: 72,325
- **Image Size**: 100x100 pixels
- **Training Set**: 54,072 images
- **Test Set**: 18,150 images
- **Classes**: 105 fruits and vegetables

---

##  Technologies Used

- **Language**: Python
- **Libraries**:
  - TensorFlow & Keras
  - NumPy
  - Matplotlib
  - PIL (Pillow)
  - OS & Shutil

---

##  Features

Convolutional Neural Network (CNN)  
Misclassified Image Detection & Sorting  
Predict custom single/multiple images  
Training & Validation Accuracy Graph  
Model Save/Reload Support

---

##  How to Run

1.  Install Dependencies:
```bash
pip install -r requirements.txt
```

2.  Train the Model:
```bash
python fruit_image_recognition_project.py
```

3.  Predict Custom Images:
Uncomment and edit this line:
```python
predict_custom_images("C:/path/to/your/image_folder")
```

4.  Output Folders:
- `misclassified_results/`: Automatically organized into folders like `true_Apple_pred_Banana`
- `fruit_classifier_cnn_model.h5`: Your saved model for reuse or deployment

---

##  Model Performance

- **Final Test Accuracy**: ~XX.XX% _(fill after training completes)_
- **Epochs Trained**: 15 _(early stopping enabled)_

Example Accuracy Plot:

![Accuracy Plot](accuracy_plot.png)

---

##  File Structure

```bash
fruit_image_recognition/
├── fruit_image_recognition_project.py
├── fruit_classifier_cnn_model.h5
├── misclassified_results/
├── README.md
└── requirements.txt
```

---

##  Dataset License
- Created by Mihai Oltean, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

##  Author
Karan Singh
