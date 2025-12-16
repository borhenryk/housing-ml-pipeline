# 🏠 Housing ML Pipeline

An end-to-end machine learning pipeline for housing price prediction, built with Databricks Asset Bundles (DABs) and deployed via GitHub Actions CI/CD.

## 📁 Project Structure

```
housing_ml_project/
├── databricks.yml              # Bundle configuration
├── resources/
│   └── housing_ml_job.yml      # Job definitions
├── src/housing_ml/notebooks/
│   ├── 01_data_preparation.py  # Data loading & cleaning
│   ├── 02_feature_engineering.py # Feature creation
│   ├── 03_model_training.py    # Hyperparameter tuning & training
│   ├── 04_model_validation.py  # Model validation & promotion
│   └── 05_model_deployment.py  # Serving endpoint deployment
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions pipeline
└── README.md
```

## 🚀 Quick Start

### Prerequisites

1. **Databricks CLI** installed and configured
2. **GitHub Repository** with secrets configured
3. **Unity Catalog** access in your Databricks workspace

### Local Deployment

```bash
# Navigate to project directory
cd housing_ml_project

# Validate the bundle
databricks bundle validate -t dev

# Deploy to development
databricks bundle deploy -t dev

# Run the training pipeline
databricks bundle run housing_ml_training_job -t dev
```

### GitHub Secrets Required

Configure these secrets in your GitHub repository:

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Databricks workspace URL (e.g., `https://dbc-xxx.cloud.databricks.com`) |
| `DATABRICKS_TOKEN` | Personal Access Token for Databricks |

## 📊 Pipeline Stages

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | Data Preparation | Downloads California Housing dataset, saves to Delta table |
| 2 | Feature Engineering | Handles outliers, creates new features |
| 3 | Model Training | Hyperparameter tuning with Optuna, trains GradientBoosting model |
| 4 | Model Validation | Validates model performance, promotes to champion |
| 5 | Model Deployment | Deploys model as serving endpoint |

## 🎯 Model Performance

The trained model achieves:
- **R² Score**: ~0.83
- **RMSE**: ~0.47
- **MAE**: ~0.31

## 📡 Serving Endpoint

Once deployed, the model is available as a REST API:

```python
import requests
import json

# Example prediction request
data = {
    "dataframe_records": [{
        "median_income": 5.0,
        "house_age": 20.0,
        "avg_rooms": 6.0,
        "avg_bedrooms": 1.0,
        "population": 1000.0,
        "avg_occupancy": 3.0,
        "latitude": 34.0,
        "longitude": -118.0,
        "rooms_per_bedroom": 6.0,
        "population_density": 166.67,
        "income_per_room": 0.83,
        "is_coastal": 0
    }]
}

response = requests.post(
    f"{DATABRICKS_HOST}/serving-endpoints/housing-price-predictor/invocations",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=data
)

print(f"Predicted value: ${response.json()['predictions'][0] * 100000:,.2f}")
```

## 🔄 CI/CD Workflow

The GitHub Actions workflow provides:

1. **Validation**: Validates bundle configuration
2. **Linting**: Checks Python code quality
3. **Dev Deployment**: Deploys on `develop` branch push
4. **Staging Deployment**: Deploys on `main` branch merge
5. **Production Deployment**: Deploys to production with approval

## 🏗️ Environments

| Target | Catalog | Schema | Usage |
|--------|---------|--------|-------|
| `dev` | dbdemos_henryk | boston_housing_ml | Development & testing |
| `staging` | dbdemos_henryk | boston_housing_ml_staging | Pre-production |
| `prod` | dbdemos_henryk | boston_housing_ml_prod | Production |

## 📝 Configuration

Bundle variables in `databricks.yml`:

```yaml
variables:
  catalog: dbdemos_henryk
  schema: boston_housing_ml
  serving_endpoint_name: housing-price-predictor
```

## 🛠️ Development

### Running Locally

1. Clone the repository
2. Configure Databricks CLI: `databricks auth login --host <workspace_url>`
3. Validate: `databricks bundle validate -t dev`
4. Deploy: `databricks bundle deploy -t dev`

### Making Changes

1. Create a feature branch
2. Make changes to notebooks or configuration
3. Push to `develop` branch to test in dev
4. Create PR to `main` for production deployment

## 📄 License

MIT License - See LICENSE file for details.

