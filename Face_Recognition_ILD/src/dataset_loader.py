import os
import numpy as np
from PIL import Image

def load_dataset(dataset_path, target_size=(56, 50)):
    images, labels = [], []
    label_id = 0

    # Supported image formats
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.pgm', '.bmp', '.tiff')

    # Special handling for Faces94 dataset structure
    if "Faces94" in dataset_path:
        # Faces94 has subfolders: male, female, malestaff
        for category in ['male', 'female', 'malestaff']:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                continue
                
            # Each category has subject folders
            subjects = sorted(os.listdir(category_path))
            for subject in subjects:
                subject_path = os.path.join(category_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                    
                # Process images in subject folder
                process_subject_folder(subject_path, images, labels, label_id, valid_extensions, target_size)
                label_id += 1
    else:
        # Standard dataset structure (ORL, GT)
        subjects = sorted(os.listdir(dataset_path))
        for subject in subjects:
            subject_path = os.path.join(dataset_path, subject)
            if not os.path.isdir(subject_path):
                continue
                
            process_subject_folder(subject_path, images, labels, label_id, valid_extensions, target_size)
            label_id += 1

    if not images:
        raise ValueError(f"No valid images found in {dataset_path}")

    return np.array(images), np.array(labels)

def process_subject_folder(subject_path, images, labels, label_id, valid_extensions, target_size):
    """Helper function to process images in a subject folder"""
    for img_name in sorted(os.listdir(subject_path)):
        img_path = os.path.join(subject_path, img_name)
        
        # Skip non-image files and hidden files
        if (not img_name.lower().endswith(valid_extensions) or 
            img_name.startswith('.')):
            print(f"❌ Skipping non-image file: {img_name}")
            continue

        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
            
            images.append(img_array)
            labels.append(label_id)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            continue