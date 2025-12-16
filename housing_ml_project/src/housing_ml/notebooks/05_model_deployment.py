# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Model Deployment
# MAGIC 
# MAGIC This notebook deploys the validated model as a serving endpoint.

# COMMAND ----------

# Get parameters
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

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, 
    ServedEntityInput,
    AutoCaptureConfigInput
)
from mlflow.tracking import MlflowClient
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Validation Status

# COMMAND ----------

# Get task values from validation
try:
    validation_passed = dbutils.jobs.taskValues.get(taskKey="model_validation", key="validation_passed")
    model_version = dbutils.jobs.taskValues.get(taskKey="model_validation", key="model_version")
except:
    # Fallback for standalone testing
    validation_passed = True
    mlflow_client = MlflowClient()
    model_name = f"{catalog}.{schema}.housing_price_model"
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    model_version = versions[0].version

print(f"Validation passed: {validation_passed}")
print(f"Model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Serving Endpoint

# COMMAND ----------

if not validation_passed:
    print("⚠️ Model validation failed. Skipping deployment.")
    dbutils.notebook.exit("Skipped - validation failed")

# COMMAND ----------

# Initialize Workspace Client
w = WorkspaceClient()

model_name = f"{catalog}.{schema}.housing_price_model"

# Check if endpoint exists
existing_endpoints = [e.name for e in w.serving_endpoints.list()]

if endpoint_name in existing_endpoints:
    print(f"📝 Updating existing endpoint: {endpoint_name}")
    
    # Update endpoint with new model version
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version=str(model_version),
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
else:
    print(f"🚀 Creating new endpoint: {endpoint_name}")
    
    # Create new endpoint
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=str(model_version),
                    scale_to_zero_enabled=True,
                    workload_size="Small"
                )
            ],
            auto_capture_config=AutoCaptureConfigInput(
                catalog_name=catalog,
                schema_name=schema,
                table_name_prefix="inference_logs",
                enabled=True
            )
        )
    )

print(f"\n✅ Endpoint deployment initiated: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to be Ready

# COMMAND ----------

# Wait for endpoint to be ready (with timeout)
max_wait_time = 600  # 10 minutes
wait_interval = 30
elapsed = 0

print("Waiting for endpoint to be ready...")

while elapsed < max_wait_time:
    endpoint_info = w.serving_endpoints.get(endpoint_name)
    state = endpoint_info.state.ready
    
    if state == "READY":
        print(f"\n✅ Endpoint is ready!")
        break
    elif state == "NOT_READY":
        config_update = endpoint_info.state.config_update
        print(f"   Status: {config_update} ({elapsed}s elapsed)")
        time.sleep(wait_interval)
        elapsed += wait_interval
    else:
        print(f"   Unknown state: {state}")
        time.sleep(wait_interval)
        elapsed += wait_interval

if elapsed >= max_wait_time:
    print(f"⚠️ Timeout waiting for endpoint. Current state: {endpoint_info.state}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Endpoint

# COMMAND ----------

import pandas as pd
import requests
import json

# Get endpoint URL
endpoint_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints/{endpoint_name}/invocations"

# Test data
test_data = {
    "dataframe_records": [
        {
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
        }
    ]
}

# Get token for authentication
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

try:
    response = requests.post(endpoint_url, headers=headers, json=test_data)
    response.raise_for_status()
    
    prediction = response.json()
    print("✅ Endpoint test successful!")
    print(f"   Input: median_income=5.0, house_age=20, avg_rooms=6.0")
    print(f"   Predicted house value: ${prediction['predictions'][0] * 100000:,.2f}")
except Exception as e:
    print(f"⚠️ Endpoint test failed (may still be initializing): {e}")

# COMMAND ----------

# Final summary
print("\n" + "=" * 60)
print("DEPLOYMENT SUMMARY")
print("=" * 60)
print(f"Model: {model_name}")
print(f"Version: {model_version}")
print(f"Endpoint: {endpoint_name}")
print(f"Inference Logging: Enabled")
print("=" * 60)
print("\n✅ Model deployment complete!")

