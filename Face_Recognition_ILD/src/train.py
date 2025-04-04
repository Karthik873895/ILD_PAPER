import os
import numpy as np
import joblib
from pathlib import Path  # For proper path handling
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

from dataset_loader import load_dataset
from feature_extraction import extract_features

# Ensure consistent dimensions for feature extraction
def ensure_consistent_dimensions(X):
    """Ensure all images have consistent dimensions (height, width)."""
    if len(X.shape) == 4:  # If has channel dimension (batch, height, width, channels)
        X = np.squeeze(X, axis=-1)
    elif len(X.shape) == 3 and X.shape[-1] == 1:  # If single channel
        X = np.squeeze(X, axis=-1)
    return X

def main():
    # Set up paths
    script_dir = Path(__file__).resolve().parent  # Get script directory
    project_root = script_dir.parent  # Get project root directory
    dataset_dir = project_root / "datasets"  # Dataset folder path
    models_dir = project_root / "models"  # Models save directory

    # Ensure models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets with verification
    datasets = {
        "ORL": dataset_dir / "ORL",
        "GT": dataset_dir / "GT",
        "Faces94": dataset_dir / "Faces94"
    }

    X_list, y_list = [], []
    for name, path in datasets.items():
        if not path.exists():
            print(f"❌ ERROR: Dataset path not found -> {path}")
            return  # Stop execution if dataset is missing
        print(f"✅ Loading {name} dataset...")
        X, y = load_dataset(str(path))
        X_list.append(ensure_consistent_dimensions(X))
        y_list.append(y)

    # Concatenate datasets
    print("🔄 Concatenating datasets...")
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    # Extract features
    print("🛠 Extracting features (this may take a while)...")
    X_features = np.array([extract_features(img) for img in X])

    # Standardize features
    print("📏 Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Apply PCA for dimensionality reduction
    print("🔽 Applying PCA...")
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"✅ Reduced to {X_pca.shape[1]} principal components")

    # Split data into training and test sets
    print("📂 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights for imbalanced data
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Hyperparameter tuning with SVM
    print("⚙️ Training SVM with GridSearchCV...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }
    svm = GridSearchCV(
        SVC(class_weight=class_weight_dict, probability=True),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    svm.fit(X_train, y_train)
    print(f"✅ Best parameters: {svm.best_params_}")

    # Evaluate model
    print("📊 Evaluating model...")
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== 📑 Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Save models and transformers
    print("💾 Saving models...")
    joblib.dump(svm.best_estimator_, str(models_dir / "ild_svm_model.pkl"))
    joblib.dump(scaler, str(models_dir / "scaler.pkl"))
    joblib.dump(pca, str(models_dir / "pca.pkl"))
    
    print(f"✅ Models saved to: {models_dir}")
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
