import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import json
import os
from model_train import DiagnosticModel
from recommender import RecommenderEngine
from ingest_nppes import fetch_providers_by_taxonomy
from parse_insurance_coverage import parse_insurance_coverage

class TestNeuroTriage(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 1. Train Diagnostic Model (Real Data)
        print("Training Diagnostic Model with Real Data...")
        cls.model = DiagnosticModel()
        # Load real data from 'data' directory
        data = cls.model.load_real_data("data")
        if data.empty:
            print("Real data empty, falling back to simulation.")
            data = cls.model.simulate_data(n_samples=500)
        cls.model.train(data)
        
        # 2. Initialize Recommender
        cls.recommender = RecommenderEngine()
        
        # 3. Use Real CMS Sample File
        cls.mock_insurance_file = "data/cms_sample.json"
        if not os.path.exists(cls.mock_insurance_file):
            print("CMS sample not found, creating mock.")
            cls.mock_insurance_file = "mock_insurance.json"
            with open(cls.mock_insurance_file, 'w') as f:
                data = [{"npi": "1111111111", "in_network": True}]
                json.dump(data, f)

    @classmethod
    def tearDownClass(cls):
        if cls.mock_insurance_file == "mock_insurance.json" and os.path.exists(cls.mock_insurance_file):
            os.remove(cls.mock_insurance_file)

    def test_end_to_end_flow(self):
        import traceback
        try:
            print("\nRunning End-to-End Test with Real Data Integration...")
            
            # Input: "Chest pain" (mapped to code E_53 if possible, or just passed as feature)
            # Note: In real DDXPlus, symptoms are codes. We'll assume 'E_53' is a valid code for this test.
            # If the model was trained on real data, it expects these codes.
            
            patient_profile = {
                'AGE': 45,
                'SEX': 'M',
                'zip': '10001',
                'symptoms': {'E_53': 1} # E_53 is often Chest Pain in DDXPlus examples
            }
            
            # Step 1: Diagnosis
            # Flatten symptoms for model input
            model_input = {
                'AGE': patient_profile['AGE'],
                'SEX': patient_profile['SEX'],
                **patient_profile['symptoms']
            }
            
            diagnoses = self.model.predict_proba(model_input)
            if diagnoses:
                top_diagnosis = diagnoses[0][0]
                print(f"Predicted Diagnosis: {top_diagnosis}")
            else:
                top_diagnosis = "Unknown"
                print("No diagnosis predicted.")
            
            # Step 2: Map to Taxonomy (Mocked Mapping for MVP)
            taxonomy_map = {
                'Acute Coronary Syndrome': '207RC0000X',
                'Myocarditis': '207RC0000X',
                'Pneumonia': '207RP1001X',
                'Appendicitis': '207RG0100X',
                'Viral Pharyngitis': '207Q00000X',
                'URTI': '207Q00000X' # Common in DDXPlus
            }
            target_taxonomy = taxonomy_map.get(top_diagnosis, '207Q00000X')
            print(f"Mapped Taxonomy: {target_taxonomy}")
            
            # Step 3: Fetch Providers (Mocked API Call)
            # We mock fetch_providers_by_taxonomy to return a list that includes the NPI from the CMS sample
            # CMS Sample NPI: 1111111111
            with patch('ingest_nppes.requests.get') as mock_get:
                # Mock response data
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "results": [
                        {
                            "number": "1111111111", # In CMS Sample
                            "basic": {"first_name": "Real", "last_name": "Doctor", "organization_name": "CMS Group"},
                            "taxonomies": [{"code": target_taxonomy, "desc": "Specialist"}],
                            "addresses": [{"city": "New York", "state": "NY", "postal_code": "10001"}]
                        },
                        {
                            "number": "9999999999", # Not in CMS Sample
                            "basic": {"first_name": "Fake", "last_name": "Doctor", "organization_name": "Nowhere"},
                            "taxonomies": [{"code": target_taxonomy, "desc": "Specialist"}],
                            "addresses": [{"city": "New York", "state": "NY", "postal_code": "10001"}]
                        }
                    ]
                }
                mock_get.return_value = mock_response
                
                providers_df = fetch_providers_by_taxonomy(target_taxonomy, "10001")
                print(f"Fetched {len(providers_df)} providers from NPPES (Mocked).")
                
            # Step 4: Filter by Insurance (Real CMS Sample Parsing)
            providers_list = providers_df.to_dict('records')
            target_npis = set(p['npi'] for p in providers_list)
            
            insurance_map = {}
            # Parse the real CMS sample file
            for record in parse_insurance_coverage(self.mock_insurance_file, target_npis):
                insurance_map[record['npi']] = record.get('in_network', False)
                
            # Annotate providers
            for p in providers_list:
                p['in_network'] = insurance_map.get(p['npi'], False)
                p['gender'] = 'M' # Mock
                
            # Step 5: Recommendation / Ranking
            ranked_providers = self.recommender.rank_providers(patient_profile, providers_list)
            
            print(f"Ranked Providers: {len(ranked_providers)}")
            for p in ranked_providers:
                print(f" - Dr. {p['last_name']} (NPI: {p['npi']}) | Score: {p['affinity_score']} | In-Network: {p['in_network']}")
                
            # Verification
            # 1. Should contain Dr. Real (1111111111) - In Network (found in CMS sample)
            self.assertTrue(any(p['npi'] == "1111111111" for p in ranked_providers))
            
            # 2. Should NOT contain Dr. Fake (9999999999) - Not in file
            self.assertFalse(any(p['npi'] == "9999999999" for p in ranked_providers))
            
            print("Test Passed!")
        except Exception:
            traceback.print_exc()
            raise

if __name__ == "__main__":
    unittest.main()
