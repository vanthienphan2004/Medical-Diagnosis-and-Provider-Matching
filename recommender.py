from typing import Dict, Any, List
import math

class RecommenderEngine:
    def __init__(self):
        pass

    def calculate_distance(self, zip1: str, zip2: str) -> float:
        """
        Calculates distance between two zip codes.
        For MVP, this is a mock implementation.
        Returns 1.0 if match, 10.0 if close (same prefix), 100.0 otherwise.
        """
        if zip1 == zip2:
            return 1.0
        elif zip1[:3] == zip2[:3]:
            return 10.0
        else:
            return 100.0

    def calculate_affinity(self, patient_profile: Dict[str, Any], provider_profile: Dict[str, Any]) -> float:
        """
        Calculates the affinity score between a patient and a provider.
        
        Logic: Score = (GenderMatch * 0.2) + (1/Distance * 0.3) + (InNetwork * 0.5)
        InNetwork is a hard filter (score = 0 if False).
        """
        
        # 1. In-Network Check (Hard Filter)
        in_network = provider_profile.get('in_network', False)
        if not in_network:
            return 0.0
        
        # 2. Gender Match
        # Assuming patient_profile['sex'] is 'M'/'F' and provider has 'gender' or similar
        # Note: NPPES data usually has 'provider_gender_code' (M/F)
        patient_sex = patient_profile.get('sex')
        provider_sex = provider_profile.get('gender') # Normalized key
        
        gender_match_score = 1.0 if patient_sex == provider_sex else 0.0
        
        # 3. Distance
        patient_zip = patient_profile.get('zip')
        provider_zip = provider_profile.get('zip')
        
        distance = self.calculate_distance(str(patient_zip), str(provider_zip))
        # Avoid division by zero
        if distance <= 0:
            distance = 0.1
            
        distance_score = 1.0 / distance
        
        # Weighted Sum
        # Weights: Gender=0.2, Distance=0.3, Network=0.5
        # Since Network is 1.0 (if true), the max score is 0.2 + 0.3 + 0.5 = 1.0
        
        final_score = (gender_match_score * 0.2) + (distance_score * 0.3) + (1.0 * 0.5)
        
        return round(final_score, 4)

    def rank_providers(self, patient_profile: Dict[str, Any], providers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks a list of providers based on affinity score.
        """
        scored_providers = []
        for provider in providers:
            score = self.calculate_affinity(patient_profile, provider)
            if score > 0:
                provider_with_score = provider.copy()
                provider_with_score['affinity_score'] = score
                scored_providers.append(provider_with_score)
        
        # Sort by score descending
        scored_providers.sort(key=lambda x: x['affinity_score'], reverse=True)
        
        return scored_providers

if __name__ == "__main__":
    engine = RecommenderEngine()
    
    patient = {'sex': 'M', 'zip': '10001'}
    
    providers = [
        {'name': 'Dr. A', 'gender': 'M', 'zip': '10001', 'in_network': True}, # Perfect match
        {'name': 'Dr. B', 'gender': 'F', 'zip': '10001', 'in_network': True}, # Gender mismatch
        {'name': 'Dr. C', 'gender': 'M', 'zip': '90210', 'in_network': True}, # Far away
        {'name': 'Dr. D', 'gender': 'M', 'zip': '10001', 'in_network': False} # Out of network
    ]
    
    ranked = engine.rank_providers(patient, providers)
    for p in ranked:
        print(f"{p['name']}: {p['affinity_score']}")
