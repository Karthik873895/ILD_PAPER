import os
import numpy as np
import joblib
from pathlib import Path
from feature_extraction import extract_features
from dataset_loader import load_dataset

def test_model():
    # Get the absolute path to the models directory
    script_dir = Path(__file__).resolve().parent  # Ensures correct path resolution
    models_dir = script_dir.parent / "models"

    # Load models
    try:
        svm = joblib.load(models_dir / "ild_svm_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        pca = joblib.load(models_dir / "pca.pkl")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}\nLooking in: {models_dir}")
        print("Ensure you've run train.py first to generate the model files.")
        return

    # Load test data
    dataset_path = script_dir.parent / "datasets" / "ORL"  # Fix relative path issue
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return

    print("Loading test dataset...")
    X_test, y_test = load_dataset(str(dataset_path))  # Convert Path to string if needed

    # Preprocess test data
    X_features = np.array([extract_features(img) for img in X_test])
    X_scaled = scaler.transform(X_features)
    X_pca = pca.transform(X_scaled)

    # Make predictions
    y_pred = svm.predict(X_pca)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_model()
