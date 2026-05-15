import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import requests
import io

def retrain():
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))

    # Features used in app.py
    # ['tenure', 'PhoneService', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']
    
    # Preprocessing
    # PhoneService: No -> 0, Yes -> 1
    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
    
    # Contract: Month-to-month -> 0, One year -> 1, Two year -> 2
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    
    # PaperlessBilling: No -> 0, Yes -> 1
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    
    # PaymentMethod: 
    # Bank transfer (automatic) -> 0
    # Credit card (automatic) -> 1
    # Electronic check -> 2
    # Mailed check -> 3
    payment_map = {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
    df['PaymentMethod'] = df['PaymentMethod'].map(payment_map)
    
    # Churn: No -> 0, Yes -> 1
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    features = ['tenure', 'PhoneService', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']
    X = df[features]
    y = df['Churn']

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("Saving model to randomforest.pkl...")
    with open('randomforest.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Retraining complete!")

if __name__ == "__main__":
    retrain()
