import streamlit as st
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import timm
from PIL import Image
from scipy.stats import mode
import gdown

# Define the function to crop black borders
def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image[y:y+h, x:x+w]
    else:
        cropped_image = image
    
    return cropped_image

# Define preprocessing functions
def min_pooling(image, alpha=4, beta=-4, sigma=10, gamma=128):
    image = image.astype(np.float32)
    gaussian_filtered = cv2.GaussianBlur(image, (0, 0), sigma)
    preprocessed_image = alpha * image + beta * gaussian_filtered + gamma
    preprocessed_image = np.clip(preprocessed_image, 0, 255).astype(np.uint8)
    return preprocessed_image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image_rgb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)
    clahe_image = cv2.merge((r_clahe, g_clahe, b_clahe))
    clahe_image_bgr = cv2.cvtColor(clahe_image, cv2.COLOR_RGB2BGR)
    return clahe_image_bgr

def apply_lab_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    clahe_lab_image = cv2.merge((l_clahe, a, b))
    clahe_image_bgr = cv2.cvtColor(clahe_lab_image, cv2.COLOR_Lab2BGR)
    return clahe_image_bgr

def apply_maxgreengsc_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = image[:, :, 1]
    max_pixel = np.max(image, axis=2)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_clahe = clahe.apply(gray)
    green_clahe = clahe.apply(green)
    max_pixel_clahe = clahe.apply(max_pixel)
    clahe_image = cv2.merge((gray_clahe, green_clahe, max_pixel_clahe))
    return clahe_image

# Function to preprocess an image using all methods
def preprocess_image(image):
    resized_image = cv2.resize(image, (500, 500))
    cropped_image = crop_black_borders(resized_image)
    preprocessed_images = {
        'no_preprocessing': cropped_image,
        'rgb_clahe': apply_clahe(cropped_image),
        'min_pooling': min_pooling(cropped_image),
        'lab_clahe': apply_lab_clahe(cropped_image),
        'maxgreengsc_clahe': apply_maxgreengsc_clahe(cropped_image)
    }
    return preprocessed_images

def predict_with_model(model, image, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        score = torch.softmax(output, dim=1).cpu().numpy()
    return pred.cpu().numpy()[0], score

# Function to download models from Google Drive
def download_model(url, output_path):
    gdown.download(url, output_path, quiet=False)

# URLs for Google Drive links
model_urls = {
    'model_no_preprocessing': 'https://drive.google.com/uc?export=download&id=1w_x2nqd-UNZJd17MSha_CDvyJbZNsknv',
    'model_rgb_clahe': 'https://drive.google.com/uc?export=download&id=1j231qvGMS1E2gWxRDcKql7ap-qKU7jBn',
    'model_min_pooling': 'https://drive.google.com/uc?export=download&id=1p4yKgrgqmP3qaacgjHeH0Ds-1eAgVqL1',
    'model_lab_clahe': 'https://drive.google.com/uc?export=download&id=1lSp_mf9YSMez7O7nelZEaSNhXOqXRDi6',
    'model_maxgreengsc_clahe': 'https://drive.google.com/uc?export=download&id=1B75d18Mn6t99iQ-nrfIaiSz2dYi3KrW3'
}

# Download model files
model_paths = [f'model_{key}.pth' for key in model_urls.keys()]
for url, path in zip(model_urls.values(), model_paths):
    download_model(url, path)

# Load the models
def load_model(model_path, device):
    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3)  # Adjust the number of classes if necessary
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
    model = model.to(device)
    return model

# Initialize models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = [load_model(path, device) for path in model_paths]

# Function to predict with all models
def predict_with_ensemble(image_path, models, device):
    image = np.array(Image.open(image_path))
    preprocessed_images = preprocess_image(image)
    all_preds = []
    for model, (name, preprocessed_image) in zip(models, preprocessed_images.items()):
        pred, _ = predict_with_model(model, preprocessed_image, device)
        all_preds.append(pred)
    final_pred_hard, _ = mode(all_preds)
    final_pred_hard = final_pred_hard.flatten()[0]
    return final_pred_hard

# Recommendations and Notes
def recommendations(pred):
    if pred == 0:
        return "Your retina appears healthy. Continue regular eye check-ups."
    elif pred == 1:
        return "You have non-referable diabetic retinopathy. Consult with an ophthalmologist for further evaluation and management."
    elif pred == 2:
        return "You have referable diabetic retinopathy. It is important to seek immediate medical attention from a specialist."
    return "Unable to determine recommendation."

def doctor_notes(pred):
    if pred == 0:
        return "The patient shows no signs of diabetic retinopathy. Regular monitoring is advised."
    elif pred == 1:
        return "The patient has non-referable diabetic retinopathy. Consider follow-up and possible interventions."
    elif pred == 2:
        return "The patient has referable diabetic retinopathy. Immediate consultation with a specialist is recommended."
    return "No data available."

def main():
    st.title("Diabetic Retinopathy Detection")
    st.markdown(
        """
        <style>
        .main {background-color: #f0f2f6; padding: 20px;}
        .card {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);}
        .recommendation {color: #1e90ff;}
        .notes {color: #ff6347;}
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Processing..."):
                image_path = uploaded_file.name
                image.save(image_path)
                pred = predict_with_ensemble(image_path, models, device)
                st.write("**Prediction:**", "Healthy" if pred == 0 else "Non-Referable DR" if pred == 1 else "Referable DR")
                
                st.markdown(f"### Recommendations")
                st.markdown(f"<div class='recommendation'>{recommendations(pred)}</div>", unsafe_allow_html=True)

                st.markdown(f"### Notes for Doctor")
                st.markdown(f"<div class='notes'>{doctor_notes(pred)}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

