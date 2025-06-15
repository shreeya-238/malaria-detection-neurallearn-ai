🦠 Malaria Detection using CNN
This project uses a Convolutional Neural Network (CNN) to detect whether a given cell image is parasitized (P) or uninfected (U). The dataset is sourced from TensorFlow Datasets and the model is built using TensorFlow and Keras.

📂 Dataset
Source: malaria from tensorflow_datasets.
Content: RGB images of blood cell slides labeled as:
  0 → Parasitized
  1 → Uninfected

📁 Data Preprocessing
Images resized to 224x224.
Normalized to pixel range [0, 1].
Dataset split into:
  80% training
  10% validation
  10% testing

🧠 CNN Architecture
  InputLayer (224x224x3)
        ↓
  Conv2D + BatchNorm + MaxPool
        ↓
  Conv2D + BatchNorm + MaxPool
        ↓
  GlobalAveragePooling2D
        ↓
  Dense (1000) + Dropout
        ↓
  Dense (100) + Dropout
        ↓
  Dense (1) with Sigmoid Activation
  Used ReLU activation in hidden layers.
  Final layer uses Sigmoid (for binary classification).
  Used GlobalAveragePooling2D to reduce overfitting (instead of Flatten).

⚙️ Compilation & Training
  Loss Function: BinaryCrossentropy
  Optimizer: Adam
  Metric: BinaryAccuracy

🛑 Callbacks Used
    ModelCheckpoint: Saves the best model (best_model.h5) based on validation loss.
    EarlyStopping: Stops training when val_loss does not improve for 5 epochs and restores the best weights.

📈 Model Performance
  After training, model evaluation was done on the test dataset.
  ✅ Final Evaluation Results:
      Test Loss:     0.1043
      Test Accuracy: 96.61%
  💡 The model achieved high accuracy, correctly identifying parasitized vs uninfected cells in over 96% of cases on unseen data.

🔍 Visualization
    A random batch of 9 test images is shown with predictions.
    Each image displays:
    Ground Truth Label (P = Parasitized, U = Uninfected)
    Model Prediction (also shown as P or U)

📊 Training Graphs
  Loss Curve:
    Shows training and validation loss over epochs.
  Accuracy Curve:
    Shows training and validation accuracy.
  These plots help identify if the model is overfitting or underfitting.

💾 Model Saving
    📥 If you're using Google Colab:
    The model file is temporarily saved. To keep it permanently:

✅ Requirements
    Python ≥ 3.
    TensorFlow ≥ 2.0
    matplotlib
    numpy
    tensorflow-datasets

📌 Key Learnings
CNNs are powerful for image classifications
GlobalAveragePooling2D helps reduce overfitting.
TensorFlow Datasets makes data loading fast and easy.
ModelCheckpoint and EarlyStopping prevent wasted training time and improve results.
