import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_PATH = '/content/drive/MyDrive/testing/model_weights/my_model.h5'
TEST_DIR = '/content/drive/MyDrive/binary_classification'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = None  # Will be inferred

def load_test_data(test_dir, img_height, img_width, batch_size):
    """
    Load test data using tf.data API with optimized performance.
    """
    test_ds = image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical', #'categorical',  # Use 'binary' for binary classification
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False
    )
    
    class_names = test_ds.class_names
    global NUM_CLASSES
    NUM_CLASSES = len(class_names)
    
    # Optimize the dataset performance
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return test_ds, class_names

def load_trained_model(model_path):
    """
    Load the trained Keras model.
    """
    model = load_model(model_path)
    return model

def evaluate_model(model, test_ds):
    """
    Evaluate the model on the test dataset and return predictions and true labels.
    """
    # Get predictions
    predictions = model.predict(test_ds)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = np.concatenate([y for x, y in test_ds], axis=0)
    true_classes = np.argmax(true_classes, axis=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    
    return loss, accuracy, true_classes, predicted_classes, predictions

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plot the confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(y_true, y_pred, class_labels):
    """
    Print and plot the classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='viridis')
    plt.title('Classification Report')
    plt.show()
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

def plot_roc_auc(y_true, predictions, class_labels):
    """
    Plot ROC curves for each class and compute AUC.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    y_true_binarized = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_labels[i]} (AUC = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load test data
    test_ds, class_labels = load_test_data(TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    
    # Load model
    model = load_trained_model(MODEL_PATH)
    
    # Evaluate model
    loss, accuracy, y_true, y_pred, predictions = evaluate_model(model, test_ds)
    
    # Print Loss and Accuracy
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")
    
    # Classification Report
    plot_classification_report(y_true, y_pred, class_labels)
    
    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_labels)
    
    # ROC AUC
    if NUM_CLASSES == 2:
        plot_roc_auc(y_true, predictions, class_labels)
    else:
        print("ROC AUC plot is typically for binary classification. Skipping.")
    
if __name__ == "__main__":
    import pandas as pd  # Imported here to avoid unnecessary dependency if not plotting classification report
    main()
