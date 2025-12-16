# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Preparation
# MAGIC 
# MAGIC This notebook downloads the California Housing dataset and saves it as a Delta table.

# COMMAND ----------

# MAGIC %pip install scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog", "dbdemos_henryk")
dbutils.widgets.text("schema", "boston_housing_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Rename columns for clarity
column_mapping = {
    'MedInc': 'median_income',
    'HouseAge': 'house_age',
    'AveRooms': 'avg_rooms',
    'AveBedrms': 'avg_bedrooms',
    'Population': 'population',
    'AveOccup': 'avg_occupancy',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'MedHouseVal': 'median_house_value'
}
df = df.rename(columns=column_mapping)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# COMMAND ----------

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Save as Delta table
table_name = f"{catalog}.{schema}.housing_data"
spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✅ Data saved to: {table_name}")
print(f"   Total rows: {spark_df.count()}")

# COMMAND ----------

# Verify the data
display(spark.sql(f"SELECT * FROM {table_name} LIMIT 10"))

# COMMAND ----------

# Save data statistics for next step
stats = df.describe().to_dict()
dbutils.jobs.taskValues.set(key="row_count", value=len(df))
dbutils.jobs.taskValues.set(key="column_count", value=len(df.columns))
print(f"✅ Data preparation complete!")

