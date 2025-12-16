# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature Engineering
# MAGIC 
# MAGIC This notebook performs feature engineering including outlier handling and feature creation.

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
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load raw data
table_name = f"{catalog}.{schema}.housing_data"
df = spark.sql(f"SELECT * FROM {table_name}")

print(f"Loaded {df.count()} rows from {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outlier Detection and Handling

# COMMAND ----------

# Define outlier handling function
def clip_outliers_spark(df, column, lower_pct=0.01, upper_pct=0.99):
    """Clip outliers using percentile-based thresholds"""
    quantiles = df.approxQuantile(column, [lower_pct, upper_pct], 0.01)
    lower_bound, upper_bound = quantiles[0], quantiles[1]
    
    return df.withColumn(
        column,
        F.when(F.col(column) < lower_bound, lower_bound)
         .when(F.col(column) > upper_bound, upper_bound)
         .otherwise(F.col(column))
    )

# Apply outlier clipping to relevant columns
outlier_columns = ['avg_rooms', 'avg_bedrooms', 'avg_occupancy', 'population']

df_cleaned = df
for col in outlier_columns:
    df_cleaned = clip_outliers_spark(df_cleaned, col)
    print(f"Clipped outliers in: {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Create new features
df_features = df_cleaned \
    .withColumn("rooms_per_bedroom", F.col("avg_rooms") / F.col("avg_bedrooms")) \
    .withColumn("population_density", F.col("population") / F.col("avg_rooms")) \
    .withColumn("income_per_room", F.col("median_income") / F.col("avg_rooms")) \
    .withColumn("is_coastal", 
        F.when((F.col("longitude") < -122) & (F.col("latitude") > 37), 1).otherwise(0)) \
    .withColumn("house_age_category",
        F.when(F.col("house_age") <= 10, "new")
         .when(F.col("house_age") <= 30, "mid")
         .otherwise("old"))

print("Created new features:")
print("  - rooms_per_bedroom")
print("  - population_density")
print("  - income_per_room")
print("  - is_coastal")
print("  - house_age_category")

# COMMAND ----------

# Save engineered features
features_table = f"{catalog}.{schema}.housing_features"
df_features.write.format("delta").mode("overwrite").saveAsTable(features_table)

print(f"✅ Features saved to: {features_table}")
print(f"   Total rows: {df_features.count()}")
print(f"   Total columns: {len(df_features.columns)}")

# COMMAND ----------

# Display sample
display(df_features.limit(10))

# COMMAND ----------

# Pass info to next task
dbutils.jobs.taskValues.set(key="features_table", value=features_table)
print("✅ Feature engineering complete!")

