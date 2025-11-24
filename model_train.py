import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Dict, Tuple, Any
import json
import ast
import os

class DiagnosticModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.all_symptoms = []
        
    def load_real_data(self, data_dir: str = "data") -> pd.DataFrame:
        """
        Loads real DDXPlus data from the data directory.
        """
        # 1. Load Evidence (Symptoms)
        evidences_path = os.path.join(data_dir, "release_evidences.json")
        if os.path.exists(evidences_path):
            with open(evidences_path, 'r') as f:
                evidences = json.load(f)
                # Filter for binary symptoms (ignoring multi-choice for MVP simplicity if needed, 
                # but let's just take all keys that are symptoms)
                self.all_symptoms = list(evidences.keys())
        else:
            print("Warning: release_evidences.json not found. Using mock symptoms.")
            self.all_symptoms = ['chest_pain', 'cough', 'fever', 'abdominal_pain', 'headache']

        # 2. Load Train Sample
        train_path = os.path.join(data_dir, "train_sample.csv")
        if not os.path.exists(train_path):
            print("train_sample.csv not found. Simulating data.")
            return self.simulate_data()
            
        print(f"Loading real data from {train_path}...")
        try:
            # Skip bad lines (e.g. truncated last line)
            df = pd.read_csv(train_path, on_bad_lines='skip')
            
            # Check if we have enough data and classes
            if df.empty or df['PATHOLOGY'].nunique() < 2:
                print("Not enough real data or classes found. Simulating data.")
                return self.simulate_data()
                
        except Exception as e:
            print(f"Error reading CSV: {e}. Simulating data.")
            return self.simulate_data()

        # 3. Process Symptoms
        # The CSV has a 'SYMPTOMS' column which is a string representation of a list
        # e.g. "['symptom_a', 'symptom_b']"
        
        # Create a binary matrix for symptoms
        # This can be slow for large data, but fine for sample
        
        # Initialize dictionary for new columns
        symptom_data = {sym: [] for sym in self.all_symptoms}
        
        valid_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Parse symptoms
                symptoms_str = row['SYMPTOMS']
                if isinstance(symptoms_str, str):
                    # Handle potential formatting issues
                    try:
                        current_symptoms = ast.literal_eval(symptoms_str)
                    except:
                        current_symptoms = []
                else:
                    current_symptoms = []
                
                # Fill binary vector
                for sym in self.all_symptoms:
                    symptom_data[sym].append(1 if sym in current_symptoms else 0)
                
                valid_rows.append(idx)
            except Exception as e:
                # Skip malformed rows
                continue
                
        # Filter df to valid rows
        df = df.loc[valid_rows].copy()
        
        # Add symptom columns
        for sym, values in symptom_data.items():
            df[sym] = values
            
        return df

    def simulate_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Simulates the DDXPlus dataset structure for testing/training.
        """
        np.random.seed(42)
        
        # Features
        ages = np.random.randint(18, 90, n_samples)
        sexes = np.random.choice(['M', 'F'], n_samples)
        
        # Mock Symptoms (Binary vector simulation)
        # Let's assume a few key symptoms
        if not self.all_symptoms:
             self.all_symptoms = ['chest_pain', 'cough', 'fever', 'abdominal_pain', 'headache']
             
        symptom_data = {sym: np.random.randint(0, 2, n_samples) for sym in self.all_symptoms}
        
        # Target Pathology (Simplified)
        # Logic: Chest pain -> likely Heart related, Cough/Fever -> Respiratory, etc.
        pathologies = []
        for i in range(n_samples):
            if 'chest_pain' in symptom_data and symptom_data['chest_pain'][i] == 1:
                pathologies.append('Acute Coronary Syndrome')
            elif 'cough' in symptom_data and 'fever' in symptom_data and symptom_data['cough'][i] == 1 and symptom_data['fever'][i] == 1:
                pathologies.append('Pneumonia')
            elif 'abdominal_pain' in symptom_data and symptom_data['abdominal_pain'][i] == 1:
                pathologies.append('Appendicitis')
            else:
                pathologies.append('Viral Pharyngitis')
        
        # Force diversity if needed
        if n_samples >= 4:
            pathologies[0] = 'Acute Coronary Syndrome'
            pathologies[1] = 'Pneumonia'
            pathologies[2] = 'Appendicitis'
            pathologies[3] = 'Viral Pharyngitis'
                
        df = pd.DataFrame({
            'AGE': ages,
            'SEX': sexes,
            'PATHOLOGY': pathologies,
            **symptom_data
        })
        
        return df

    def train(self, df: pd.DataFrame):
        """
        Trains the XGBoost model.
        """
        # Drop non-feature columns
        # Keep AGE, SEX, and all symptom columns
        # Drop metadata columns from DDXPlus if present
        drop_cols = ['PATHOLOGY', 'DIFFERENTIAL_DIAGNOSIS', 'INITIAL_EVIDENCE', 'SYMPTOMS']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df['PATHOLOGY']
        
        self.feature_columns = X.columns.tolist()
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Preprocessing
        categorical_features = ['SEX']
        numeric_features = ['AGE']
        # Ensure these exist
        if 'SEX' not in X.columns:
            X['SEX'] = 'M' # Fallback
        if 'AGE' not in X.columns:
            X['AGE'] = 30 # Fallback
            
        # All other columns are binary symptoms (numeric)
        symptom_features = [col for col in X.columns if col not in ['AGE', 'SEX']]
        numeric_features.extend(symptom_features)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                n_estimators=100
            ))
        ])
        
        self.model.fit(X, y_encoded)
        print("Model training complete.")

    def predict_proba(self, input_data: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns the top k differential diagnoses with probabilities.
        """
        if not self.model:
            raise ValueError("Model not trained yet.")
            
        # Convert input dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all feature columns exist (fill missing with 0 for symptoms)
        for col in self.feature_columns:
            if col not in df.columns:
                if col == 'SEX':
                    df[col] = 'M' # Default or handle appropriately
                elif col == 'AGE':
                    df[col] = 30 # Default
                else:
                    df[col] = 0
                    
        # Predict
        probs = self.model.predict_proba(df)[0]
        
        # Map back to labels
        class_indices = np.argsort(probs)[::-1][:top_k]
        top_diagnoses = []
        
        for idx in class_indices:
            label = self.label_encoder.inverse_transform([idx])[0]
            prob = float(probs[idx])
            top_diagnoses.append((label, prob))
            
        return top_diagnoses

if __name__ == "__main__":
    model = DiagnosticModel()
    # Try loading real data first
    data = model.load_real_data()
    if data.empty:
        data = model.simulate_data()
        
    model.train(data)
    
    test_case = {
        'AGE': 45,
        'SEX': 'M',
        'chest_pain': 1, # Make sure this matches a real symptom name if using real data
        'E_53': 1 # Example code from DDXPlus if applicable, otherwise use names
    }
    
    print("Prediction for test case:", test_case)
    print(model.predict_proba(test_case))
