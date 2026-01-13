import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import numpy as np
import json
import sys
import os
import scipy.stats as st

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def format_predictions(predictions):
    """Format PDF predictions for display"""
    results = []

    print(f"Debug: Predictions shape: {predictions.shape}", file=sys.stderr)
    print(f"Debug: Predictions content: {predictions}", file=sys.stderr)

    # Now predictions should be (n_samples, 2) - no more 3D arrays!
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        print(f"Debug: Processing prediction {i}: {pred}", file=sys.stderr)

        # Extract probabilities
        if isinstance(pred, np.ndarray) and len(pred) >= 2:
            p_defective = float(pred[0])
            p_non_defective = float(pred[1])
        else:
            print(f"Warning: Unexpected prediction format for {i}: {pred}", file=sys.stderr)
            p_defective = 0.0
            p_non_defective = 0.0

        results.append({
            'p_defective': p_defective,
            'p_non_defective': p_non_defective
        })

        print(f"Debug: Class {i} - P(Defective): {p_defective}, P(Non-Defective): {p_non_defective}", file=sys.stderr)

    return results

def format_results_for_comparison(class_names, base_data, head_data):
    """Format results as tables comparing BEFORE and AFTER for each class"""
    # First, calculate percentage changes for all classes to determine sorting order
    file_changes = []

    for i, class_name in enumerate(class_names):
        if i < len(base_data) and i < len(head_data):
            base_defective = base_data[i]['p_defective']
            base_non_defective = base_data[i]['p_non_defective']
            head_defective = head_data[i]['p_defective']
            head_non_defective = head_data[i]['p_non_defective']

            # Calculate percentage changes
            if base_defective != 0:
                defective_change = ((head_defective - base_defective) / abs(base_defective)) * 100
            else:
                defective_change = 0 if head_defective == 0 else float('inf')

            if base_non_defective != 0:
                non_defective_change = ((head_non_defective - base_non_defective) / abs(base_non_defective)) * 100
            else:
                non_defective_change = 0 if head_non_defective == 0 else float('inf')

            # Use the maximum absolute change for sorting
            max_change = max(abs(defective_change), abs(non_defective_change))

            file_changes.append({
                'index': i,
                'class_name': class_name,
                'max_change': max_change,
                'defective_change': defective_change,
                'non_defective_change': non_defective_change,
                'base_defective': base_defective,
                'base_non_defective': base_non_defective,
                'head_defective': head_defective,
                'head_non_defective': head_non_defective
            })
        else:
            file_changes.append({
                'index': i,
                'class_name': class_name,
                'max_change': 0,
                'error': True
            })

    # Sort classes by maximum percentage change in descending order
    file_changes.sort(key=lambda x: x['max_change'], reverse=True)

    # Generate output with sorted classes
    output = ["## 📊 Bug Prediction Analysis\n"]

    for file_data in file_changes:
        class_name = file_data['class_name']
        output.append(f"#### Class: `{class_name}`\n")

        if 'error' in file_data:
            output.append("| Status |")
            output.append("|--------|")
            output.append("| Error: Prediction data not available |")
            output.append("")
        else:
            base_defective = file_data['base_defective']
            base_non_defective = file_data['base_non_defective']
            head_defective = file_data['head_defective']
            head_non_defective = file_data['head_non_defective']
            defective_change = file_data['defective_change']
            non_defective_change = file_data['non_defective_change']

            # Format percentage change values
            def format_change(change_val):
                if change_val == float('inf'):
                    return "∞%"
                elif change_val == float('-inf'):
                    return "-∞%"
                else:
                    return f"{change_val:+.2f}%"

            before = "Defective" if base_defective > base_non_defective else "Non-Defective"
            after = "Defective" if head_defective > head_non_defective else "Non-Defective"

            output.append("Outcome: " + before + " -> " + after)

            # Create table with 4 columns
            output.append("| Metric | BEFORE PR | AFTER PR | % Change |")
            output.append("|--------|-----------|----------|----------|")
            output.append(f"| PDF(Defective \\| Reconstruction Error) | {base_defective:.4f} | {head_defective:.4f} | {format_change(defective_change)} |")
            output.append(f"| PDF(Non-Defective \\| Reconstruction Error) | {base_non_defective:.4f} | {head_non_defective:.4f} | {format_change(non_defective_change)} |")
            output.append("")

    return "\n".join(output)

def format_results(class_names, prediction_data):
    """Format results with probability values (for individual calls)"""
    results = []

    for i, class_name in enumerate(class_names):
        if i < len(prediction_data):
            p_defective = prediction_data[i]['p_defective']
            p_non_defective = prediction_data[i]['p_non_defective']

            results.append({
                'class': class_name,
                'p_defective': p_defective,
                'p_non_defective': p_non_defective
            })
        else:
            results.append({
                'class': class_name,
                'error': 'No prediction available'
            })

    return results

def get_distribution_class(dist_name):
    """Get the distribution class (not frozen) from scipy.stats"""
    if dist_name is None:
        return None

    try:
        dist_class = getattr(st, dist_name)
        return dist_class
    except Exception as e:
        return None

def load_trained_model(model_dir="trained_model"):
    """Load the pre-trained model"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained model not found at {model_dir}. Please ensure the model is trained and saved.")

    # Load metadata from JSON
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load REPD classifier parameters from JSON
    classifier_params_path = os.path.join(model_dir, "classifier_params.json")
    if not os.path.exists(classifier_params_path):
        raise FileNotFoundError(f"Classifier parameters not found at {classifier_params_path}")

    with open(classifier_params_path, 'r') as f:
        classifier_params = json.load(f)

    # Recreate the autoencoder with saved architecture
    autoencoder = AutoEncoder(
        metadata['architecture'], 
        metadata['learning_rate'], 
        metadata['epochs'], 
        metadata['batch_size']
    )

    # Load the saved autoencoder weights
    autoencoder_path = os.path.join(model_dir, "autoencoder")
    autoencoder.load(autoencoder_path)

    # Recreate REPD classifier
    classifier = REPD(autoencoder)

    # Non-defective distribution
    classifier.dnd = get_distribution_class(classifier_params.get('dnd_name'))
    classifier.dnd_pa = tuple(classifier_params.get('dnd_params', []))

    # Defective distribution  
    classifier.dd = get_distribution_class(classifier_params.get('dd_name'))
    classifier.dd_pa = tuple(classifier_params.get('dd_params', []))

    # Check if distributions were created successfully
    if classifier.dnd is None:
        raise ValueError("Failed to get non-defective distribution class")
    if classifier.dd is None:
        raise ValueError("Failed to get defective distribution class")

    return classifier

def predict(features_file, model_dir="trained_model"):
    """Make predictions using pre-trained model"""

    classifier = load_trained_model(model_dir)

    # Load test data
    try:
        df_test = pd.read_csv(features_file)
    except Exception as e:
        print(f"Error: Failed to read CSV file {features_file}: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Check if CSV has data rows (more than just header)
    if len(df_test) == 0:
        print("Error: No classes to analyze in the CSV file.", file=sys.stderr)
        return []

    # Verify required columns
    expected_columns = [
        'project_name', 'version', 'class_name', 'wmc', 'rfc', 'loc', 'max_cc', 'avg_cc',
        'cbo', 'ca', 'ce', 'ic', 'cbm', 'lcom', 'lcom3', 'dit', 'noc', 'mfa',
        'npm', 'dam', 'moa', 'cam', 'amc', 'bug'
    ]
    if not all(col in df_test.columns for col in expected_columns):
        missing = [col for col in expected_columns if col not in df_test.columns]
        print(f"Error: CSV file is missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Extract class names and numerical features
    class_names = df_test["class_name"].values
    # Select numerical metrics, excluding non-feature columns
    feature_columns = [
        'wmc', 'rfc', 'loc', 'max_cc', 'avg_cc', 'cbo', 'ca', 'ce', 'ic', 'cbm',
        'lcom', 'lcom3', 'dit', 'noc', 'mfa', 'npm', 'dam', 'moa', 'cam', 'amc'
    ]
    X_test = df_test[feature_columns].values

    print(f"Debug: Processing {len(class_names)} classes", file=sys.stderr)
    print(f"Debug: Class names: {class_names}", file=sys.stderr)
    print(f"Debug: X_test shape: {X_test.shape}", file=sys.stderr)

    # Make predictions (PDF values)
    try:
        pdf_predictions = classifier.predict(X_test)
    except Exception as e:
        print(f"Error: Prediction failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"Debug: Predictions shape: {pdf_predictions.shape}", file=sys.stderr)
    print(f"Debug: Predictions type: {type(pdf_predictions)}", file=sys.stderr)

    # Format predictions for display
    prediction_data = format_predictions(pdf_predictions)

    # Format and return results
    results = format_results(class_names, prediction_data)

    # Close the session
    classifier.dim_reduction_model.close()

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_features.csv>")
        print("Make sure the trained model exists in the 'trained_model' directory.")
        sys.exit(1)

    features_csv_path = sys.argv[1]
    results = predict(features_csv_path)

    # For command line usage, print individual results
    for result in results:
        if 'error' in result:
            print(f"Class: {result['class']}")
            print(f"Error: {result['error']}")
        else:
            print(f"Class: {result['class']}")
            print(f"PDF(Defective | Reconstruction Error): {result['p_defective']:.4f}")
            print(f"PDF(Non-Defective | Reconstruction Error): {result['p_non_defective']:.4f}")
        print()
