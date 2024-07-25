# ------------------------------------------------------- 
# Requirements
# ------------------------------------------------------- 
from fastapi import FastAPI,HTTPException
import numpy as np
from pydantic import BaseModel
import joblib
from pathlib import Path


# ------------------------------------------------------- 
# App
# ------------------------------------------------------- 
app = FastAPI()

# ------------------------------------------------------- 
# Utils
# ------------------------------------------------------- 
class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def load_model(model_path):
    return joblib.load(model_path)


def preprocess(data: CustomerData):
    # Exemple de prétraitement - ajustez-le en fonction des besoins de votre modèle
    feature_order = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]

    categorical_features = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod'
    ]

    # Convertir l'objet Pydantic en dictionnaire
    data_dict = data.dict()

    # Encodeur fictif (à remplacer par votre encodeur réel)
    encoded_features = []
    for feature in feature_order:
        if feature in categorical_features:
            # Encoder la caractéristique catégorielle (exemple de transformation)
            encoded_feature = encode_categorical_feature(feature, data_dict[feature])
            encoded_features.extend(encoded_feature)
        else:
            encoded_features.append(data_dict[feature])
    
    return np.array(encoded_features).reshape(1, -1)

def encode_categorical_feature(feature, value):
    # Exemple d'encodage pour une caractéristique catégorielle
    # Remplacez-le par votre méthode d'encodage réelle (par exemple, One-Hot Encoding)
    encoding_map = {
        'Yes': 1,
        'No': 0,
        # Ajoutez ici toutes les valeurs possibles pour chaque caractéristique catégorielle
    }
    return [encoding_map.get(value, -1)]  # Retourne -1 si la valeur n'est pas trouvée (à ajuster)

# ------------------------------------------------------- 
# Load the model on app setup
# ------------------------------------------------------- 
model_path = Path(__file__).parent / "best_model.pkl"
model = load_model(model_path)
# ------------------------------------------------------- 
# First route
# ------------------------------------------------------- 
@app.get("/")
def api_info():
    return {"info": "Welcome to the churn prediction API"}

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 
@app.post("/predict")
async def predict(data: CustomerData):
    try:
        # Prétraiter les données
        processed_data = preprocess(data)
        
        # Faire la prédiction
        prediction = model.predict(processed_data)
        
        # Retourner la prédiction
        return {
            "Churn": "Yes" if prediction[0] else "No"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
