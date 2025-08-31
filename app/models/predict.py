"""
Contains the logic for making clinical outcome predictions by loading
and using pre-trained machine learning models and preprocessing objects.
"""
import joblib
import pandas as pd
from pathlib import Path # <-- FIX: Import pathlib for robust path handling

try:
    APP_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = APP_DIR / "saved_models"

    copd_classifier = joblib.load(MODEL_DIR / 'copd_classifier.joblib')
    alt_regressor = joblib.load(MODEL_DIR / 'alt_regressor.joblib')
    copd_label_encoder = joblib.load(MODEL_DIR / 'copd_label_encoder.joblib')
    top_copd_features = joblib.load(MODEL_DIR / 'top_copd_features.joblib')
    top_alt_features = joblib.load(MODEL_DIR / 'top_alt_features.joblib')
    encoded_columns = joblib.load(MODEL_DIR / 'encoded_columns.joblib')
    MODELS_LOADED = True
    print("--> [Predictor] All models and preprocessing objects loaded successfully.")

except FileNotFoundError as e:
    MODELS_LOADED = False
    print(f"--> [Predictor] WARNING: Model files not found. Error: {e}")
    print(f"--> [Predictor] Looked in directory: {MODEL_DIR.resolve()}")
    print("--> Please ensure the 'saved_models' directory exists at the project root.")


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
            "error": "Prediction models are not loaded. Please check the container logs for errors."
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