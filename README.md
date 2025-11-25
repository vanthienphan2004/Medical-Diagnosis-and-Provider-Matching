# NeuroTriage: US-Based Clinical Routing System

A production-grade system that diagnoses symptoms, maps them to US Provider Taxonomies, and filters providers based on insurance data.

## Features

- **Symptom-Based Diagnosis**: Uses XGBoost ML model trained on DDXPlus dataset
- **Provider Matching**: Maps diagnoses to NUCC Healthcare Provider Taxonomy Codes
- **Insurance Filtering**: Streams large CMS Machine-Readable Files (MRFs) using `ijson`
- **Provider Ranking**: "Habit Match" algorithm scoring based on gender, distance, and network status
- **HIPAA Compliant**: No PHI logging or storage

## Architecture

```
Input Symptoms → Diagnostic Model → Taxonomy Mapping → Provider Fetch (NPPES) → Insurance Filter → Ranking → Output
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Medical-Diagnosis-and-Provider-Matching
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**:
   ```bash
   python download_real_data.py
   ```

## Usage

### Run Tests
```bash
python test_suite.py
```

### Generate Visualizations
```bash
python visualize.py
```

This creates `neurotriage_visualization.png` with:
- Pathology distribution in training data
- Age distribution by pathology
- Gender distribution
- Model prediction confidence
- Provider affinity scores by scenario
- System architecture flow diagram

### Train Model
```bash
python model_train.py
```

### Example: Fetch Providers
```python
from ingest_nppes import fetch_providers_by_taxonomy

# Fetch cardiologists in New York
providers = fetch_providers_by_taxonomy("207RC0000X", "10001")
print(providers)
```

## Project Structure

```
├── design_spec.md              # System design and mapping strategy
├── download_real_data.py       # Downloads DDXPlus and CMS samples
├── ingest_nppes.py            # NPPES API integration
├── model_train.py             # XGBoost diagnostic model
├── parse_insurance_coverage.py # Streaming JSON parser for MRFs
├── recommender.py             # Provider ranking engine
├── test_suite.py              # End-to-end validation
└── data/                      # Downloaded datasets (gitignored)
```

## Data Sources

- **DDXPlus**: Symptom-pathology dataset from [HuggingFace](https://huggingface.co/datasets/aai530-group6/ddxplus)
- **NPPES**: Provider data from [CMS NPPES API](https://npiregistry.cms.hhs.gov/api-page)
- **Insurance**: CMS Machine-Readable Files from [price-transparency-guide](https://github.com/CMSgov/price-transparency-guide)

## Key Components

### 1. Diagnostic Model (`model_train.py`)
- Trains XGBoost classifier on DDXPlus data
- Returns top-5 differential diagnoses with probabilities
- Supports real data and simulated fallback

### 2. NPPES Integration (`ingest_nppes.py`)
- Fetches providers by taxonomy code and zip
- Returns structured DataFrame

### 3. Insurance Parser (`parse_insurance_coverage.py`)
- Streams 100GB+ JSON files using `ijson`
- Filters by NPI for in-network status

### 4. Recommender (`recommender.py`)
- Scores providers: `(GenderMatch × 0.2) + (1/Distance × 0.3) + (InNetwork × 0.5)`
- Hard filter on in-network status

## Testing

The test suite validates:
- ✅ Real data loading (DDXPlus)
- ✅ Diagnosis prediction
- ✅ Taxonomy mapping
- ✅ Provider fetching (mocked NPPES API)
- ✅ Insurance filtering (real CMS sample)
- ✅ Provider ranking

## HIPAA Compliance

- No PHI is logged or stored
- All patient data processed in-memory only
- Logs contain only taxonomy codes and pathology categories

## License

MIT

## Contributing

Pull requests welcome! Please ensure tests pass before submitting.

## Contact

For questions or issues, please open a GitHub issue.
