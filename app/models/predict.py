"""
Contains the logic for making clinical outcome predictions by loading
and using pre-trained machine learning models and preprocessing objects.
"""
import joblib
import pandas as pd
import os

try:
    # This assumes the script is in /app/models/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

    copd_classifier = joblib.load(os.path.join(MODEL_DIR, 'copd_classifier.joblib'))
    alt_regressor = joblib.load(os.path.join(MODEL_DIR, 'alt_regressor.joblib'))
    copd_label_encoder = joblib.load(os.path.join(MODEL_DIR, 'copd_label_encoder.joblib'))
    top_copd_features = joblib.load(os.path.join(MODEL_DIR, 'top_copd_features.joblib'))
    top_alt_features = joblib.load(os.path.join(MODEL_DIR, 'top_alt_features.joblib'))
    encoded_columns = joblib.load(os.path.join(MODEL_DIR, 'encoded_columns.joblib'))
    MODELS_LOADED = True
    print("--> [Predictor] All models and preprocessing objects loaded successfully.")
except FileNotFoundError:
    MODELS_LOADED = False
    # Use a more helpful error message showing the exact path it checked
    expected_path = os.path.abspath(MODEL_DIR)
    print(f"--> [Predictor] WARNING: Model files not found in the expected directory: {expected_path}")
    print("--> Please ensure the 'saved_models' directory exists at the project root and run `train_model.py`.")


def predict_patient_outcomes(patient_features: dict) -> dict:
    """
    Predicts clinical outcomes using loaded machine learning models.

    This function preprocesses the input features to match the exact format used
    during training, then runs inference with the appropriate models.

    Args:
        patient_features: A dictionary of raw patient data provided by the agent.

    Returns:
        A dictionary containing the predicted outcomes.
    """
    if not MODELS_LOADED:
        return {
            "error": "Prediction models are not loaded. Please run the training script first."
        }

    print(f"--> [Predictor] Received raw features for prediction: {patient_features}")

    # 1. Convert the input dictionary to a pandas DataFrame
    input_df = pd.DataFrame([patient_features])

    # 2. One-Hot Encode the input data using the same columns as the training script
    input_encoded = pd.get_dummies(input_df)

    # 3. Align columns with the training data's full feature set
    input_aligned = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # 4. Predict COPD
    copd_features_df = input_aligned[top_copd_features]
    copd_prediction_encoded = copd_classifier.predict(copd_features_df)
    copd_prediction = copd_label_encoder.inverse_transform(copd_prediction_encoded)[0]

    # 5. Predict ALT
    alt_features_df = input_aligned[top_alt_features]
    alt_prediction = alt_regressor.predict(alt_features_df)[0]

    result = {
        "predicted_chronic_obstructive_pulmonary_disease": str(copd_prediction),
        "predicted_alanine_aminotransferase": round(float(alt_prediction), 4)
    }

    print(f"--> [Predictor] Returning final predictions: {result}")
    return result

