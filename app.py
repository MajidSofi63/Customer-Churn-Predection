import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border: none;
    }
    .css-1r6slb0 {
        padding: 2rem 1rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #60a5fa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load the Model ---
@st.cache_resource
def load_model():
    return pickle.load(open('randomforest.pkl', 'rb'))

model = load_model()

# Define the columns for user input
columns = ['tenure', 'PhoneService', 'Contract',
           'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']

def predict_churn(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data], columns=columns)
    # Make predictions using the loaded model
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:, 1][0]
    return prediction, probability

def main():
    st.title("Customer Churn Prediction App")
    st.write("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Customer Profile")
        with st.container():
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with the service.")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.1)
            
            st.divider()
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                phone_service = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                contract = st.selectbox("Contract Type", [0, 1, 2], format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
            
            with sub_col2:
                paperless_billing = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                payment_method = st.selectbox("Payment Method", [0, 1, 2, 3], format_func=lambda x: [
                    "Bank transfer (auto)", 
                    "Credit card (auto)", 
                    "Electronic check", 
                    "Mailed check"
                ][x])

        predict_btn = st.button("Analyze Risk Profile")

    with col2:
        st.subheader("Analysis Result")
        
        input_data = {
            'tenure': tenure,
            'PhoneService': phone_service,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges
        }

        if predict_btn:
            prediction, probability = predict_churn(input_data)
            
            # Gauge-like display using progress bar
            st.markdown(f"#### Churn Probability: `{probability:.1%}`")
            if probability > 0.6:
                st.error("🚨 **HIGH RISK**: This customer is very likely to churn.")
                st.progress(probability, text="Critical Risk")
            elif probability > 0.4:
                st.warning("⚠️ **ELEVATED RISK**: Moderate risk of churn detected.")
                st.progress(probability, text="Moderate Risk")
            else:
                st.success("✅ **LOW RISK**: This customer is likely to stay.")
                st.progress(probability, text="Healthy Account")

            st.write("---")
            st.subheader("💡 Retention Strategy")
            if prediction == 1 or probability > 0.4:
                st.info("""
                **Recommendations:**
                - Offer a loyalty discount or bundle upgrade.
                - Proactively reach out for a service health check.
                - Consider moving the customer to a longer-term contract.
                """)
            else:
                st.info("""
                **Maintenance:**
                - Continue regular service updates.
                - Monitor for any sudden usage changes.
                """)
        else:
            st.info("Fill out the profile on the left and click 'Analyze Risk Profile' to see the results.")

    

if __name__ == '__main__':
    main()
