import ijson
import json
from typing import Generator, Set, Dict, Any

def parse_insurance_coverage(file_path: str, target_npis: Set[str]) -> Generator[Dict[str, Any], None, None]:
    """
    Streams a CMS Machine Readable File (JSON) and yields records
    matching the provided set of NPIs.
    
    For the CMS schema, we look for 'provider_groups' which contain 'npi' lists.
    If an NPI is found, we consider them 'in_network'.
    """
    try:
        with open(file_path, 'rb') as f:
            # The CMS schema typically has a 'provider_references' list
            # We want to stream items from 'provider_references'
            # Each item has 'provider_groups' -> [{'npi': [...]}]
            
            # Try parsing 'provider_references.item'
            parser = ijson.items(f, 'provider_references.item')
            
            for ref in parser:
                groups = ref.get('provider_groups', [])
                for group in groups:
                    npis = group.get('npi', [])
                    # Check if any target NPI is in this group
                    for npi in npis:
                        npi_str = str(npi)
                        if npi_str in target_npis:
                            print(f"DEBUG: Found NPI {npi_str} in file.")
                            yield {
                                "npi": npi_str,
                                "in_network": True,
                                "tin": group.get('tin', {}).get('value')
                            }
                            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error parsing insurance file: {e}")

