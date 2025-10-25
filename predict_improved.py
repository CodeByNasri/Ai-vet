# predict_improved.py - Enhanced prediction script for lumpy skin disease detection
import os
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------- CONFIG ----------
MODEL_PATH = "final_improved_model.keras"      # Your new trained model
FALLBACK_MODEL = "mobilenetv2_model.keras"     # Fallback to original model
IMG_SIZE = (224, 224)
DATA_SPLIT_TRAIN = "data_split/train"
# ----------------------------

def load_model():
    """Load the best available model"""
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"Loaded improved model: {MODEL_PATH}")
        return model, "efficientnet"
    elif os.path.exists(FALLBACK_MODEL):
        model = keras.models.load_model(FALLBACK_MODEL)
        print(f"Loaded fallback model: {FALLBACK_MODEL}")
        return model, "mobilenet"
    else:
        raise FileNotFoundError(f"No model found. Expected {MODEL_PATH} or {FALLBACK_MODEL}")

# Load model and determine class names
model, model_type = load_model()

if os.path.isdir(DATA_SPLIT_TRAIN):
    class_names = sorted([d for d in os.listdir(DATA_SPLIT_TRAIN) 
                         if os.path.isdir(os.path.join(DATA_SPLIT_TRAIN, d))])
    print("Class names (from data_split/train):", class_names)
else:
    class_names = ["Lumpy Skin", "Normal Skin"]
    print("Using fallback class names:", class_names)

def prepare_image(path, model_type="efficientnet"):
    """Enhanced image preprocessing"""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    
    # Use appropriate preprocessing based on model type
    if model_type == "efficientnet":
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    else:  # mobilenet
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_confidence(image_path, model_type="efficientnet"):
    """Predict with confidence and additional metrics"""
    x = prepare_image(image_path, model_type)
    preds = model.predict(x, verbose=0)
    probs = preds[0]
    
    # Get prediction
    idx = int(np.argmax(probs))
    predicted_class = class_names[idx]
    confidence = float(probs[idx])
    
    # Calculate additional metrics
    second_highest = np.sort(probs)[-2]
    confidence_gap = confidence - second_highest
    
    return predicted_class, confidence, probs, confidence_gap

def predict_single_enhanced(image_path):
    """Enhanced single image prediction with detailed output"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    class_name, confidence, probs, gap = predict_with_confidence(image_path, model_type)
    
    print(f"\n=== Prediction Results ===")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Confidence Gap: {gap*100:.2f}%")
    
    print(f"\nClass Probabilities:")
    for i, (class_name_prob, prob) in enumerate(zip(class_names, probs)):
        status = "✓" if i == np.argmax(probs) else " "
        print(f"  {status} {class_name_prob}: {prob*100:.2f}%")
    
    # Confidence interpretation
    if confidence > 0.9:
        confidence_level = "Very High"
    elif confidence > 0.8:
        confidence_level = "High"
    elif confidence > 0.7:
        confidence_level = "Medium"
    elif confidence > 0.6:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    print(f"\nConfidence Level: {confidence_level}")
    
    return class_name, confidence, probs

def predict_folder_enhanced(folder_path, save_csv=True, csv_name="enhanced_predictions.csv"):
    """Enhanced folder prediction with detailed analysis"""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return
    
    rows = []
    predictions = []
    confidences = []
    
    print(f"\n=== Batch Prediction: {folder_path} ===")
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_files:
        print("No image files found in the folder.")
        return
    
    print(f"Found {len(image_files)} image files")
    
    for i, fname in enumerate(sorted(image_files)):
        path = os.path.join(folder_path, fname)
        try:
            class_name, confidence, probs, gap = predict_with_confidence(path, model_type)
            
            # Store results
            predictions.append(class_name)
            confidences.append(confidence)
            
            # Create detailed row for CSV
            row = [fname, class_name, f"{confidence:.6f}", f"{gap:.6f}"]
            row.extend([f"{p:.6f}" for p in probs])
            rows.append(row)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Processed {i + 1}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue
    
    # Save detailed CSV
    if save_csv and rows:
        header = ["filename", "predicted_class", "confidence", "confidence_gap"]
        header.extend([f"prob_{c}" for c in class_names])
        
        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nDetailed predictions saved to: {csv_name}")
    
    # Summary statistics
    print(f"\n=== Batch Prediction Summary ===")
    unique, counts = np.unique(predictions, return_counts=True)
    for class_name, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"{class_name}: {count} images ({percentage:.1f}%)")
    
    avg_confidence = np.mean(confidences)
    print(f"Average Confidence: {avg_confidence*100:.2f}%")
    print(f"High Confidence (>80%): {sum(1 for c in confidences if c > 0.8)} images")
    print(f"Low Confidence (<60%): {sum(1 for c in confidences if c < 0.6)} images")

def evaluate_test_set():
    """Evaluate model on test set with detailed metrics"""
    test_path = os.path.join("data_split", "test")
    if not os.path.isdir(test_path):
        print("Test directory not found. Skipping evaluation.")
        return
    
    print("\n=== Test Set Evaluation ===")
    
    true_labels = []
    predicted_labels = []
    confidences = []
    
    for class_name in class_names:
        class_path = os.path.join(test_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            path = os.path.join(class_path, fname)
            try:
                pred_class, confidence, _, _ = predict_with_confidence(path, model_type)
                true_labels.append(class_name)
                predicted_labels.append(pred_class)
                confidences.append(confidence)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue
    
    if not true_labels:
        print("No test images found.")
        return
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to: confusion_matrix.png")
    
    # Accuracy by class
    print("\nAccuracy by Class:")
    for class_name in class_names:
        class_mask = np.array(true_labels) == class_name
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(predicted_labels)[class_mask] == class_name)
            print(f"{class_name}: {class_acc*100:.2f}%")

def create_gradcam_enhanced(image_path, out_path="gradcam_enhanced.jpg"):
    """Enhanced Grad-CAM visualization"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    img = prepare_image(image_path, model_type)
    
    # Find the last convolutional layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D) or 'conv' in layer.name.lower():
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        print("No convolutional layer found for Grad-CAM.")
        return
    
    print(f"Using layer: {last_conv_layer_name}")
    
    # Create Grad-CAM model
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    
    # Create visualization
    orig = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE).convert("RGBA")
    orig_rgba = orig.convert("RGBA")
    blended = Image.blend(orig_rgba, heatmap_img, alpha=0.4)
    
    # Add prediction info to image
    class_name, confidence, _, _ = predict_with_confidence(image_path, model_type)
    blended.save(out_path)
    print(f"Enhanced Grad-CAM saved to: {out_path}")
    print(f"Prediction: {class_name} ({confidence*100:.2f}%)")

# --------------------------
# Main execution examples
# --------------------------
if __name__ == "__main__":
    print("=== Enhanced Lumpy Skin Disease Detector ===")
    print(f"Model Type: {model_type.upper()}")
    print(f"Classes: {class_names}")
    
    # Example 1: Single image prediction
    test_image = r"test_images\web_image_13.jpg"
    if os.path.exists(test_image):
        predict_single_enhanced(test_image)
    else:
        print(f"Test image not found: {test_image}")
    
    # Example 2: Evaluate on test set
    evaluate_test_set()
    
    # Example 3: Enhanced folder prediction
    # predict_folder_enhanced(r"data\Lumpy Skin", save_csv=True, csv_name="lumpy_enhanced_preds.csv")
    
    # Example 4: Enhanced Grad-CAM
    # create_gradcam_enhanced(test_image, "gradcam_enhanced.jpg")
    
    print("\n=== Ready for enhanced predictions ===")
    print("Available functions:")
    print("- predict_single_enhanced(image_path)")
    print("- predict_folder_enhanced(folder_path)")
    print("- evaluate_test_set()")
    print("- create_gradcam_enhanced(image_path)")
