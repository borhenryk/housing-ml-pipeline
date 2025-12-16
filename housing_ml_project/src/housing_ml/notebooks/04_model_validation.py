# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Model Validation
# MAGIC 
# MAGIC This notebook validates the trained model and sets it as champion if it meets the criteria.

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog", "dbdemos_henryk")
dbutils.widgets.text("schema", "boston_housing_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Model Info

# COMMAND ----------

# Get task values from previous step
try:
    run_id = dbutils.jobs.taskValues.get(taskKey="model_training", key="run_id")
    model_name = dbutils.jobs.taskValues.get(taskKey="model_training", key="model_name")
    test_r2 = dbutils.jobs.taskValues.get(taskKey="model_training", key="test_r2")
except:
    # Fallback for standalone testing
    model_name = f"{catalog}.{schema}.housing_price_model"
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = versions[0]
    run_id = latest.run_id
    
    # Get metrics from run
    run = client.get_run(run_id)
    test_r2 = run.data.metrics.get("test_r2", 0)

print(f"Model: {model_name}")
print(f"Run ID: {run_id}")
print(f"Test R²: {test_r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Criteria

# COMMAND ----------

# Define validation criteria
MIN_R2_THRESHOLD = 0.7
MIN_IMPROVEMENT = 0.01

# Initialize client
client = MlflowClient()

# Get current model version
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = versions[0]

print(f"Latest model version: {latest_version.version}")
print(f"Status: {latest_version.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Checks

# COMMAND ----------

validation_passed = True
validation_messages = []

# Check 1: R² threshold
if test_r2 >= MIN_R2_THRESHOLD:
    validation_messages.append(f"✅ R² ({test_r2:.4f}) >= threshold ({MIN_R2_THRESHOLD})")
else:
    validation_passed = False
    validation_messages.append(f"❌ R² ({test_r2:.4f}) < threshold ({MIN_R2_THRESHOLD})")

# Check 2: Compare with previous champion (if exists)
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
    champion_run = client.get_run(champion.run_id)
    champion_r2 = champion_run.data.metrics.get("test_r2", 0)
    
    improvement = test_r2 - champion_r2
    if improvement >= MIN_IMPROVEMENT:
        validation_messages.append(f"✅ Improvement ({improvement:.4f}) >= minimum ({MIN_IMPROVEMENT})")
    elif improvement >= 0:
        validation_messages.append(f"⚠️ Marginal improvement ({improvement:.4f}), proceeding anyway")
    else:
        validation_passed = False
        validation_messages.append(f"❌ New model worse than champion by {abs(improvement):.4f}")
except:
    validation_messages.append("ℹ️ No existing champion model, skipping comparison")

# Print results
print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
for msg in validation_messages:
    print(msg)
print("=" * 60)
print(f"\nOverall: {'✅ PASSED' if validation_passed else '❌ FAILED'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Champion Alias

# COMMAND ----------

if validation_passed:
    # Set champion alias
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version.version
    )
    print(f"\n✅ Model version {latest_version.version} set as 'champion'")
    
    # Also set a 'validated' alias
    client.set_registered_model_alias(
        name=model_name,
        alias="validated",
        version=latest_version.version
    )
    print(f"✅ Model version {latest_version.version} set as 'validated'")
else:
    print("\n⚠️ Model did not pass validation. Not promoting to champion.")
    # Still set a 'latest' alias for reference
    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=latest_version.version
    )
    print(f"ℹ️ Model version {latest_version.version} set as 'candidate'")

# COMMAND ----------

# Pass validation status to deployment
dbutils.jobs.taskValues.set(key="validation_passed", value=validation_passed)
dbutils.jobs.taskValues.set(key="model_version", value=latest_version.version)

print("\n✅ Model validation complete!")

