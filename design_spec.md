# Project NeuroTriage Design Specification

## 1. Dataset Schemas

### 1.1 DDXPlus (Symptom & Pathology Data)
*Source: Mila-IQIA/ddxplus*

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `pathology` | String | Target label (e.g., "Myocarditis", "GERD"). |
| `symptoms` | List/String | List of binary symptom flags or sparse vector. |
| `age` | Integer | Patient age. |
| `sex` | String | Patient sex (M/F). |
| `initial_evidence` | String | The symptom that initiated the consultation. |

### 1.2 NPPES Data Dissemination (Provider Data)
*Source: CMS*

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `NPI` | String | 10-digit National Provider Identifier. |
| `Entity Type Code` | String | `1` = Individual, `2` = Organization. |
| `Provider Last Name` | String | Last name (Type 1). |
| `Provider First Name` | String | First name (Type 1). |
| `Provider Business Practice Location Address Postal Code` | String | Zip code (5 or 9 digits). |
| `Healthcare Provider Taxonomy Code_1` | String | Primary taxonomy code (e.g., `207RC0000X`). |

## 2. Mapping Strategy

### 2.1 Pathology to Taxonomy Mapping
We will use a direct mapping dictionary for the MVP.

**Key Mappings (Examples):**
- **Cardiovascular**:
    - `Myocarditis`, `Acute Coronary Syndrome` -> `207RC0000X` (Internal Medicine: Cardiovascular Disease)
- **Respiratory**:
    - `Pneumonia`, `Asthma` -> `207RP1001X` (Internal Medicine: Pulmonary Disease)
- **Gastrointestinal**:
    - `GERD`, `Appendicitis` -> `207RG0100X` (Internal Medicine: Gastroenterology)
- **General/Primary Care**:
    - `Viral Pharyngitis`, `Influenza` -> `207Q00000X` (Family Medicine)

### 2.2 Fallback Mechanism
If a specific pathology is not found in the map, default to `207Q00000X` (Family Medicine) or `208D00000X` (General Practice).

## 3. HIPAA Compliance & Privacy

### 3.1 Data Handling
- **No PHI Storage**: Patient symptoms and demographics are processed in-memory only for the duration of the request.
- **De-identification**: Any logs must exclude `age`, `sex`, and specific `symptoms`. Log only the `pathology` category or `taxonomy_code` if needed for debugging.

### 3.2 Security
- **Input Validation**: Strict validation of input JSON to prevent injection attacks.
- **Access Control**: Ensure the API is accessible only via authorized channels (mocked for this MVP).

## 4. System Architecture

1.  **Ingestion**: Fetch Providers (NPPES) -> Filter by Taxonomy & Zip.
2.  **Diagnosis**: Input Symptoms -> XGBoost Model -> Predicted Pathology.
3.  **Mapping**: Predicted Pathology -> Taxonomy Code.
4.  **Filtering**: Filter Providers by Insurance (Mocked/Streamed).
5.  **Ranking**: Score Providers (Habit Match).
6.  **Output**: Sorted List of Providers.
