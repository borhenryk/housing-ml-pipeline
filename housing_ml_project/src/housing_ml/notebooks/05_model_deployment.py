# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Model Deployment (Skipped)
# MAGIC 
# MAGIC This notebook is a placeholder - endpoint deployment is skipped for this run.

# COMMAND ----------

# Get parameters
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
    endpoint_name = dbutils.widgets.get("endpoint_name")
except:
    dbutils.widgets.text("catalog", "dbdemos_henryk")
    dbutils.widgets.text("schema", "boston_housing_ml")
    dbutils.widgets.text("endpoint_name", "housing-price-predictor")
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
    endpoint_name = dbutils.widgets.get("endpoint_name")

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deployment Skipped
# MAGIC 
# MAGIC Endpoint deployment is disabled for this run. The model is registered in Unity Catalog and can be deployed manually if needed.

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Get model info
model_name = f"{catalog}.{schema}.housing_price_model"

try:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        latest = versions[0]
        print(f"Model is registered and ready:")
        print(f"   Name: {model_name}")
        print(f"   Version: {latest.version}")
        print(f"   Status: {latest.status}")
        print(f"\nTo deploy as an endpoint, use the Databricks UI or SDK.")
    else:
        print(f"No model versions found for {model_name}")
except Exception as e:
    print(f"Could not retrieve model info: {e}")

# COMMAND ----------

print("\nDeployment step completed (endpoint creation skipped).")
