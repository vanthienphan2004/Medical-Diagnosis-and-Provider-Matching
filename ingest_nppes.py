import requests
import pandas as pd
from typing import Optional, List, Dict, Any

API_ENDPOINT = "https://npiregistry.cms.hhs.gov/api/"

def fetch_providers_by_taxonomy(taxonomy_code: str, zip_code: str, limit: int = 50) -> pd.DataFrame:
    """
    Fetches providers from the NPPES API based on Taxonomy Code and Zip Code.

    Args:
        taxonomy_code (str): The Healthcare Provider Taxonomy Code (e.g., '207RC0000X').
        zip_code (str): The 5-digit postal code.
        limit (int): Maximum number of records to return (default 50, max 200 per API).

    Returns:
        pd.DataFrame: A DataFrame containing provider details.
    """
    params = {
        "version": "2.1",
        "postal_code": zip_code,
        "taxonomy_description": "", # API doesn't strictly filter by code in all versions, but we can try or filter post-fetch. 
                                    # Actually, the API allows 'taxonomy_description' but for codes it's trickier.
                                    # We will fetch by zip and filter by taxonomy code in memory if API doesn't support direct code.
                                    # However, let's try to pass it if possible or just fetch by zip.
                                    # Optimization: The API is limited. Fetching by ZIP is better.
        "limit": limit
    }
    
    # Note: The public NPPES API doesn't always allow direct filtering by taxonomy code easily without description.
    # We will fetch by zip and then filter client-side for the specific taxonomy code 
    # to ensure accuracy, as the API 'taxonomy_description' is fuzzy.
    
    try:
        response = requests.get(API_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        
        providers = []
        for result in results:
            # Basic info
            npi = result.get("number")
            basic = result.get("basic", {})
            first_name = basic.get("first_name")
            last_name = basic.get("last_name")
            org_name = basic.get("organization_name")
            
            # Taxonomies
            taxonomies = result.get("taxonomies", [])
            # Check if our target taxonomy exists in this provider's taxonomies
            has_taxonomy = False
            for tax in taxonomies:
                if tax.get("code") == taxonomy_code:
                    has_taxonomy = True
                    break
            
            if has_taxonomy:
                providers.append({
                    "npi": npi,
                    "first_name": first_name,
                    "last_name": last_name,
                    "org_name": org_name,
                    "taxonomy_code": taxonomy_code,
                    "city": result.get("addresses", [{}])[0].get("city"),
                    "state": result.get("addresses", [{}])[0].get("state"),
                    "zip": zip_code
                })
                
        return pd.DataFrame(providers)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from NPPES API: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test run
    # Cardiologists (207RC0000X) in New York, NY (10001)
    df = fetch_providers_by_taxonomy("207RC0000X", "10001")
    print(df.head())
