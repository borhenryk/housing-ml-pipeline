# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Model Training
# MAGIC 
# MAGIC This notebook trains a GradientBoostingRegressor model with hyperparameter optimization.

# COMMAND ----------

# MAGIC %pip install optuna --quiet
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
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load features
features_table = f"{catalog}.{schema}.housing_features"
df = spark.sql(f"SELECT * FROM {features_table}").toPandas()

# Define feature columns (exclude target and categorical columns)
feature_cols = [
    'median_income', 'house_age', 'avg_rooms', 'avg_bedrooms',
    'population', 'avg_occupancy', 'latitude', 'longitude',
    'rooms_per_bedroom', 'population_density', 'income_per_room', 'is_coastal'
]
target_col = 'median_house_value'

X = df[feature_cols]
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Features: {feature_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Optimization with Optuna

# COMMAND ----------

# Set up MLflow experiment
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/{schema}_experiment"
mlflow.set_experiment(experiment_name)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# Define Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
    return -scores.mean()

# Run optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"\n✅ Best hyperparameters found:")
for param, value in study.best_params.items():
    print(f"   {param}: {value}")
print(f"\n   Best CV MSE: {study.best_value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Final Model

# COMMAND ----------

# Train final model with best parameters
best_params = study.best_params
best_params['random_state'] = 42

with mlflow.start_run(run_name="housing_price_model") as run:
    # Train model
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2
    })
    
    # Create and log pipeline
    full_pipeline = Pipeline([
        ('scaler', scaler),
        ('regressor', model)
    ])
    
    # Log model
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, y_pred[:len(X_train)])
    
    model_name = f"{catalog}.{schema}.housing_price_model"
    mlflow.sklearn.log_model(
        full_pipeline,
        "model",
        signature=signature,
        registered_model_name=model_name,
        input_example=X_train.iloc[:5]
    )
    
    run_id = run.info.run_id
    
    print(f"\n📊 Model Performance:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"\n✅ Model registered: {model_name}")
    print(f"   Run ID: {run_id}")

# COMMAND ----------

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

display(feature_importance)

# COMMAND ----------

# Save run_id for next steps
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="model_name", value=model_name)
dbutils.jobs.taskValues.set(key="test_r2", value=r2)

print("✅ Model training complete!")

