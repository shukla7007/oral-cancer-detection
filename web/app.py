from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Load the trained model and segmentation model
model = load_model('/Users/anshulshukla/Desktop/oral cancer detection/models/cancer_detection_model.keras')
segmentation_model = load_model('/Users/anshulshukla/Desktop/oral cancer detection/models/segmentation_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def load_metrics(filename='metrics.json'):
    with open(filename, 'r') as f:
        metrics = json.load(f)
    return metrics['accuracy'], metrics['precision'], metrics['recall']

def generate_overlay(image_path):
    """Generate an overlay with magnification to highlight cancerous regions."""
    # Load and preprocess the image
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Save original dimensions for resizing
    img_resized = cv2.resize(img, (150, 150))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Get segmentation prediction
    mask = segmentation_model.predict(img_input)[0]
    mask_resized = cv2.resize(mask, original_size[::-1])  # Resize mask to original image size
    mask_binary = (mask_resized > 0.5).astype(np.uint8)  # Binarize mask to highlight cancer regions only

    # Create an overlay with transparency and magnification
    overlay_img = img.copy()
    red_overlay = np.zeros_like(img, dtype=np.uint8)
    red_overlay[:, :, 2] = 255  # Red color for highlighting

    # Magnify cancerous regions
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for cancerous region
        cancer_region = img[y:y+h, x:x+w]  # Crop the cancerous region

        # Scale up the cancerous region by 1.5x
        magnified_region = cv2.resize(cancer_region, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # Calculate the position to overlay the magnified region
        # Ensure the overlay doesn't exceed image boundaries
        overlay_x_start = max(x - (magnified_region.shape[1] - w) // 2, 0)
        overlay_y_start = max(y - (magnified_region.shape[0] - h) // 2, 0)
        overlay_x_end = min(overlay_x_start + magnified_region.shape[1], img.shape[1])
        overlay_y_end = min(overlay_y_start + magnified_region.shape[0], img.shape[0])

        # Place magnified region onto the overlay
        overlay_img[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end] = magnified_region[:overlay_y_end-overlay_y_start, :overlay_x_end-overlay_x_start]

    # Blend the overlay with the original image based on the mask
    alpha = 0.5  # Transparency factor for red highlighting
    overlay_img = cv2.addWeighted(overlay_img, 1, red_overlay, alpha, 0)
    overlay_img[mask_binary == 0] = img[mask_binary == 0]  # Keep non-cancer areas as the original image

    # Save overlay image
    overlay_filename = 'overlay_' + os.path.basename(image_path)
    overlay_path = os.path.join('/Users/anshulshukla/Desktop/oral cancer detection/web/static/images 2', overlay_filename)
    cv2.imwrite(overlay_path, overlay_img)

    return overlay_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Ensure the 'static/images 2' directory exists
        save_directory = '/Users/anshulshukla/Desktop/oral cancer detection/web/static/images 2'
        os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

        # Save the file to the 'images 2' folder
        file_path = os.path.join(save_directory, file.filename)
        file.save(file_path)
        
        # Preprocess the image
        img = preprocess_image(file_path)
        
        # Make predictions
        predictions = model.predict(img)
        
        # Flip the prediction logic
        if predictions[0][0] > 0.5:  # Assuming binary classification
            result = "Non-Cancer"  # Non-Cancer
            overlay_filename = None
        else:
            result = "Cancer"  # Cancer
            overlay_filename = generate_overlay(file_path)

        # Load accuracy, precision, and recall from metrics JSON
        accuracy, precision, recall = load_metrics()
        
        # Render the result template with prediction results and metrics
        return render_template('result.html', result=result, filename=file.filename, accuracy=accuracy, precision=precision, recall=recall, overlay_filename=overlay_filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
