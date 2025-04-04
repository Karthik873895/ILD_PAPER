import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from pathlib import Path
from dataset_loader import load_dataset
from feature_extraction import extract_features, compute_lbp, compute_elbp, compute_mbp, compute_lpq

# Create results directory if it doesn't exist
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# 1. Evaluation of Proposed Model
def evaluate_model(dataset_paths):
    """
    Evaluate the proposed model on multiple datasets
    """
    # Load trained models
    models_dir = Path("models")
    svm = joblib.load(models_dir / "ild_svm_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    pca = joblib.load(models_dir / "pca.pkl")
    
    results = {}
    
    for name, path in dataset_paths.items():
        print(f"\nEvaluating on {name} dataset...")
        X, y = load_dataset(path)
        
        # Extract features
        X_features = np.array([extract_features(img) for img in X])
        X_scaled = scaler.transform(X_features)
        X_pca = pca.transform(X_scaled)
        
        # Evaluate
        y_pred = svm.predict(X_pca)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        results[name] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm
        }
        
        # Save results
        with open(results_dir / f"{name}_evaluation.txt", "w") as f:
            f.write(f"Dataset: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y, y_pred))
            f.write("\nConfusion Matrix:\n")
            np.savetxt(f, cm, fmt='%d')
    
    return results

# 2. Graph Analysis for 3 Datasets
def plot_dataset_analysis(dataset_paths):
    """
    Generate graphs comparing the 3 datasets
    """
    dataset_stats = {}
    
    for name, path in dataset_paths.items():
        X, y = load_dataset(path)
        unique, counts = np.unique(y, return_counts=True)
        
        dataset_stats[name] = {
            'num_samples': len(X),
            'num_classes': len(unique),
            'samples_per_class': counts.mean(),
            'class_distribution': dict(zip(unique, counts))
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Samples per dataset
    plt.subplot(2, 2, 1)
    plt.bar(dataset_stats.keys(), [stats['num_samples'] for stats in dataset_stats.values()])
    plt.title("Total Samples per Dataset")
    plt.ylabel("Number of Samples")
    
    # Classes per dataset
    plt.subplot(2, 2, 2)
    plt.bar(dataset_stats.keys(), [stats['num_classes'] for stats in dataset_stats.values()])
    plt.title("Number of Classes per Dataset")
    plt.ylabel("Number of Classes")
    
    # Samples per class
    plt.subplot(2, 2, 3)
    plt.bar(dataset_stats.keys(), [stats['samples_per_class'] for stats in dataset_stats.values()])
    plt.title("Average Samples per Class")
    plt.ylabel("Samples per Class")
    
    # Class distribution example
    plt.subplot(2, 2, 4)
    for name, stats in dataset_stats.items():
        plt.plot(sorted(stats['class_distribution'].values()), label=name)
    plt.title("Class Distribution Comparison")
    plt.ylabel("Number of Samples")
    plt.xlabel("Classes (sorted)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "dataset_comparison.png")
    plt.close()
    
    return dataset_stats

# 3. RR Comparison for Different Descriptors
def compare_descriptors(dataset_paths):
    """
    Compare recognition rates for different feature descriptors
    """
    descriptor_names = ['LBP', 'ELBP', 'MBP', 'LPQ', 'ILD']
    results = {name: {} for name in dataset_paths.keys()}
    
    for dataset_name, path in dataset_paths.items():
        X, y = load_dataset(path)
        
        # Extract features for each descriptor
        features = {
            'LBP': np.array([compute_lbp(img) for img in X]),
            'ELBP': np.array([compute_elbp(img) for img in X]),
            'MBP': np.array([compute_mbp(img) for img in X]),
            'LPQ': np.array([compute_lpq(img) for img in X]),
            'ILD': np.array([extract_features(img) for img in X])
        }
        
        # Load or train models for each descriptor
        for desc_name, X_features in features.items():
            if desc_name == 'ILD':
                # Use the pre-trained model
                models_dir = Path("models")
                svm = joblib.load(models_dir / "ild_svm_model.pkl")
                scaler = joblib.load(models_dir / "scaler.pkl")
                pca = joblib.load(models_dir / "pca.pkl")
                
                X_scaled = scaler.transform(X_features)
                X_pca = pca.transform(X_scaled)
                acc = svm.score(X_pca, y)
            else:
                # Train new model for other descriptors
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_features)
                
                pca = PCA(n_components=0.95)
                X_pca = pca.fit_transform(X_scaled)
                
                svm = SVC(kernel='rbf', C=10, gamma=0.001)
                acc = np.mean(cross_val_score(svm, X_pca, y, cv=5))
            
            results[dataset_name][desc_name] = acc
    
    # Plot results
    plt.figure(figsize=(12, 6))
    x = np.arange(len(descriptor_names))
    width = 0.25
    
    for i, (dataset_name, accs) in enumerate(results.items()):
        plt.bar(x + i*width, [accs[desc] for desc in descriptor_names], 
                width, label=dataset_name)
    
    plt.xlabel('Feature Descriptors')
    plt.ylabel('Recognition Rate')
    plt.title('Recognition Rate Comparison Across Descriptors')
    plt.xticks(x + width, descriptor_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "descriptor_comparison.png")
    plt.close()
    
    # Save numerical results
    with open(results_dir / "descriptor_comparison.txt", "w") as f:
        f.write("Recognition Rate Comparison\n")
        f.write("="*30 + "\n\n")
        for dataset_name, accs in results.items():
            f.write(f"{dataset_name} Dataset:\n")
            for desc_name, acc in accs.items():
                f.write(f"{desc_name}: {acc:.4f}\n")
            f.write("\n")
    
    return results

if __name__ == "__main__":
    # Define your dataset paths
    dataset_paths = {
        "ORL": "../datasets/ORL",
        "GT": "../datasets/GT",
        "Faces94": "../datasets/Faces94"
    }
    
    print("Running evaluation of proposed model...")
    model_results = evaluate_model(dataset_paths)
    
    print("\nGenerating dataset analysis graphs...")
    dataset_stats = plot_dataset_analysis(dataset_paths)
    
    print("\nComparing different descriptors...")
    descriptor_results = compare_descriptors(dataset_paths)
    
    print("\nAll results saved in the 'results' directory!")