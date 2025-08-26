import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Helper Functions (Backend Logic) ---

def load_model_and_metadata():
    """
    Loads the trained model and its associated metadata.
    
    NOTE: You must first save your trained model and the prediction metadata
    from your Jupyter Notebook using joblib.dump() before this will work.
    For example, joblib.dump(rf, 'random_forest_model.pkl')
    """
    try:
        # Load the actual model and metadata
        model = joblib.load('random_forest_model.pkl')
        metadata = joblib.load('prediction_metadata.pkl')
        return model, metadata
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'random_forest_model.pkl' and 'prediction_metadata.pkl' are in the same directory as this app.py file.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the model files: {e}")
        st.stop()

def _prepare_data_for_prediction(raw_patient_data, model_features):
    """
    A helper function to prepare the patient data dictionary for the model.
    It explicitly maps the UI keys to the model's exact feature names to prevent mismatches.
    """
    data_for_model = {}
    
    
    for feature in model_features:
        
        if feature in raw_patient_data:
            data_for_model[feature] = raw_patient_data[feature]
        else:
            
            data_for_model[feature] = 0
            st.warning(f"Feature '{feature}' not found in patient data. Using a default value of 0.")
            
    
    return pd.DataFrame([data_for_model], columns=model_features)


def get_prediction_from_model(patient_data, model, metadata):
    """
    This function uses the real model to make a prediction on the patient data.
    """
    try:
        model_features = metadata['model_features']
        
        
        patient_df = _prepare_data_for_prediction(patient_data, model_features)
        
        
        raw_prediction = model.predict(patient_df)
        
        
        prediction_map = metadata['prediction_map']
        prediction_label = prediction_map[raw_prediction[0]]
        
        # Return the result
        return {
            'prediction': prediction_label,
            'model_used': 'Random Forest Classifier',
            'reason': 'Prediction from the trained model.'
        }
    except Exception as e:
        st.error(f"An error occurred during model prediction: {e}")
        return {
            'prediction': 'Error',
            'model_used': 'N/A',
            'reason': 'An error occurred during prediction. Please check the input data.'
        }


def safe_predict(patient_data, model, metadata):
    """
    This is the two-tiered prediction function from our previous discussion.
    It checks for high-risk symptoms first before using the model.
    """
    
    # Tier 2: Rule-Based Override
    high_risk_symptoms = metadata['suspect_features']
    for feature in high_risk_symptoms:
        # Check if the feature exists and its value is high (> 5)
        if feature in patient_data and patient_data.get(feature, 0) > 5:
            return {
                'prediction': 'High',
                'model_used': 'Rule-Based Override',
                'reason': f"High-risk symptom '{feature}' detected. Immediate professional consultation is advised."
            }
            
    # Tier 1: Model Prediction using the original patient data dictionary
    return get_prediction_from_model(patient_data, model, metadata)

# --- Streamlit Frontend ---

# Page configuration (set this as the first command)
st.set_page_config(
    page_title="CancerWise",
    page_icon="💙",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for the dark theme and styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Main title */
    h1 {
        color: #ffffff;
        text-align: center;
    }
    /* Subheader */
    h2, h3 {
        color: #c7c7c7;
        text-align: center;
    }
    /* Feature cards */
    .feature-card {
        background-color: #1e222a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        border: 1px solid #2e3440;
    }
    .feature-card h3 {
        color: #58a6ff;
        font-size: 1.25rem;
    }
    /* Get Started Button */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
        display: block;
        margin: 20px auto;
    }
    /* Disclaimer text */
    .disclaimer {
        font-size: 0.8rem;
        text-align: center;
        color: #888;
        padding-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# --- App State Management ---
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

# --- Landing Page UI ---
if not st.session_state.show_uploader:
    st.title("💙 CancerWise")
    st.markdown("<h2 style='text-align: center;'>AI-Powered Lung Cancer Risk Prediction</h2>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: center;'>Welcome to CancerWise, a privacy-first tool that analyzes your data to provide a personalized lung cancer risk assessment. Get started by uploading your report.</p>", unsafe_allow_html=True)
    
    # Feature cards in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛡️ Privacy First</h3>
            <p>Your report is processed in memory and never stored. We are committed to protecting your sensitive health data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⬆️ Secure Upload</h3>
            <p> Adjust the sliders to provide your health data and get an instant risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>✔️ Informed Decisions</h3>
            <p>Our AI provides insights based on your data, helping you have more informed conversations with your doctor.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Get Started"):
        st.session_state.show_uploader = True
        st.rerun()

# --- Questionnaire Page UI ---
else:
    st.title("Patient Information Questionnaire")
    st.markdown("Please provide your information below. All features are rated on a scale of 1 to 9, where 1 is the lowest and 9 is the highest, unless otherwise specified.")
    
    with st.form("patient_data_form"):
        # Split features into two columns for better layout
        col_left, col_right = st.columns(2)
        
        # Initialize an empty dictionary to store patient data
        patient_data_dict = {}
        
        # Define the ranges for the sliders
        feature_ranges = {
            'Age': (1, 100),
            'Gender': ('Male', 'Female'), # Special case for gender
            'Air Pollution': (1, 9),
            'Alcohol use': (1, 9),
            'Dust Allergy': (1, 9),
            'OccuPational Hazards': (1, 9),
            'Genetic Risk': (1, 9),
            'chronic Lung Disease': (1, 9),
            'Balanced Diet': (1, 9),
            'Obesity': (1, 9),
            'Smoking': (1, 9),
            'Passive Smoker': (1, 9),
            'Chest Pain': (1, 9),
            'Coughing of Blood': (1, 9),
            'Fatigue': (1, 9),
            'Weight Loss': (1, 9),
            'Shortness of Breath': (1, 9),
            'Wheezing': (1, 9),
            'Swallowing Difficulty': (1, 9),
            'Clubbing of Finger Nails': (1, 9),
            'Frequent Cold': (1, 9),
            'Dry Cough': (1, 9),
            'Snoring': (1, 9)
        }
        slider_help_texts = {'Smoking': 'This slider represents your smoking exposure. 1 is a non-smoker, and 9 is a heavy, long-term smoker.','Coughing of Blood': 'This slider measures the frequency and severity of a persistent cough. 1 is no cough, and 9 indicates a constant, severe cough.','Chest Pain': 'This slider measures the frequency and intensity of chest pain. 1 means no pain, and 9 represents very severe or constant pain.','Air Pollution': 'Use this slider to represent your regular exposure to high levels of air pollution. 1 is minimal exposure, and 9 is a high level of exposure from living or working in a polluted area.','OccuPational Hazards': 'This slider indicates your exposure to workplace lung irritants like asbestos or radon. 1 is no exposure, and 9 is significant, long-term exposure.'}
        
        
        # Iterate through features and create input widgets
        features_to_display = list(feature_ranges.keys())
        
        
        with col_left:
            for feature in features_to_display[:12]:
                if feature == 'Gender':
                    gender_input = st.selectbox(
                        'Gender', 
                        options=['Male', 'Female'],
                        help="Male = 1, Female = 2"
                    )
                    patient_data_dict[feature] = 1 if gender_input == 'Male' else 2
                else:
                    help_text = slider_help_texts.get(feature, f"Scale of {feature_ranges[feature][0]} to {feature_ranges[feature][1]}")
                    patient_data_dict[feature] = st.slider(
                        feature, 
                        min_value=feature_ranges[feature][0], 
                        max_value=feature_ranges[feature][1], 
                        value=5, # Default value
                        help=help_text
                    )
            
        with col_right:
            for feature in features_to_display[12:]:
                help_text = slider_help_texts.get(feature, f"Scale of {feature_ranges[feature][0]} to {feature_ranges[feature][1]}")
                patient_data_dict[feature] = st.slider(
                    feature, 
                    min_value=feature_ranges[feature][0], 
                    max_value=feature_ranges[feature][1], 
                    value=5, # Default value
                    help=help_text
                )
        # A form submit button must be inside the form
        submit_button = st.form_submit_button("Get Prediction")

    if submit_button:
        # Load your trained model and metadata
        model, metadata = load_model_and_metadata()
        
        # Get prediction using your two-tiered system
        with st.spinner('Analyzing data...'):
            result = safe_predict(patient_data_dict, model, metadata)
        
        # Display results
        st.subheader("Analysis Results")
        if result['prediction'] == 'High':
            st.error(f"**Predicted Risk Level: {result['prediction']}**")
        elif result['prediction'] == 'Medium':
            st.warning(f"**Predicted Risk Level: {result['prediction']}**")
        else:
            st.success(f"**Predicted Risk Level: {result['prediction']}**")

        st.info(f"**Analysis Method:** {result['model_used']}")
        st.write(f"**Reason:** {result['reason']}")
       
        # --- Feature Importance (Tier 1 only) ---
        if result['model_used'] == 'Random Forest Classifier':
            try:
                st.subheader("Top Features Influencing Risk")
                model_features = metadata['model_features']
                fi_df = pd.DataFrame({
                    'Feature': model_features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(fi_df.set_index('Feature').head(5))
            except Exception as e:
                st.warning(f"Could not calculate feature importance: {e}")

    if st.button("Back to Home"):
        st.session_state.show_uploader = False
        st.rerun()

# --- Disclaimer ---
st.markdown("<p class='disclaimer'><b>Medical Disclaimer:</b> CancerWise is not a diagnostic tool and does not provide medical advice. Consult with a qualified healthcare professional for any health concerns or before making any medical decisions.</p>", unsafe_allow_html=True)
