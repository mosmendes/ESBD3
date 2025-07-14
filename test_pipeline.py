import pytest
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, when, trim, lit, sum
import mlflow.pyfunc
import mlflow.tracking
from sklearn.metrics import accuracy_score

# --- Global/Fixture Setup (for pytest) ---
@pytest.fixture(scope="module")
def spark_session():
    """Provides a SparkSession for tests."""
    try:
        spark = SparkSession.builder \
            .appName("ChurnPredictionTests") \
            .config("spark.local.dir", os.path.join(os.getcwd(), "spark_test_tmp")) \
            .config("spark.sql.warehouse.dir", os.path.join(os.getcwd(), "spark-warehouse-test")) \
            .getOrCreate()
    except Exception: # Fallback if already running or other issue
        spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture(scope="module")
def mlflow_model():
    """Loads the MLflow model for tests."""
    #mlflow_db_path = "C:/MONICA/Estudos/mlf_data/mlflow.db" 

    tracking_uri = os.getenv('MLFLOW_DB_PATH', 'sqlite:///C:/MONICA/Estudos/mlf_data/mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("churn_prediction") 
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
            latest_run = runs[0]
            model_uri = f"runs:/{latest_run.info.run_id}/churn_model" 
            model = mlflow.pyfunc.load_model(model_uri)
            if model:
                print(f"\nMLflow model loaded from {model_uri}")
                return model
    pytest.fail("MLflow model not found or could not be loaded. Ensure the model is trained and registered.")


def clean_and_prepare_spark_df(spark_df_raw):
    """Applies cleaning and type conversion similar to 04_monitor_fairness.ipynb."""
    original_numerical_cols = [
        "Account length", "Area code", "Number vmail messages", "Total day minutes", 
        "Total day calls", "Total day charge", "Total eve minutes", "Total eve calls", 
        "Total eve charge", "Total night minutes", "Total night calls", 
        "Total night charge", "Total intl minutes", "Total intl calls", 
        "Total intl charge", "Customer service calls"
    ]
    
    original_categorical_cols = ["State", "International plan", "Voice mail plan", "Churn"]

    for col_name in original_numerical_cols:
        spark_df_raw = spark_df_raw.withColumn(
            col_name,
            when(trim(col(col_name)) == "", lit(None)) 
            .when(trim(col(col_name)).rlike("^[a-zA-Z]+$"), lit(None)) 
            .otherwise(col(col_name)) 
        )
        spark_df_raw = spark_df_raw.withColumn(col_name, col(col_name).cast(DoubleType()))

    spark_df_raw = spark_df_raw.fillna(0.0, subset=original_numerical_cols)

    for col_name in original_categorical_cols:
        if col_name in ['International plan', 'Voice mail plan']:
            spark_df_raw = spark_df_raw.withColumn(
                col_name,
                when(trim(col(col_name)) == "", "No") 
                .when(col(col_name).isNull(), "No") 
                .otherwise(col(col_name))
            )
        elif col_name == 'Churn':
             spark_df_raw = spark_df_raw.withColumn(
                col_name,
                when(trim(col(col_name)) == "", "No") 
                .when(col(col_name).isNull(), "No") 
                .otherwise(col(col_name))
            )
        else: 
            spark_df_raw = spark_df_raw.withColumn(
                col_name,
                when(trim(col(col_name)) == "", "Unknown") 
                .when(col(col_name).isNull(), "Unknown") 
                .otherwise(col(col_name))
            )
    return spark_df_raw

# --- Test Functions ---

def test_mlflow_model_loads(mlflow_model):
    """Verifies that the MLflow model can be loaded."""
    assert mlflow_model is not None

def test_sample_prediction(spark_session, mlflow_model):
    """Tests prediction with a sample Pandas DataFrame."""
    features_for_prediction = [
        "Account length", "International plan", "Number vmail messages",
        "Total day minutes", "Total day calls", "Total eve minutes",
        "Total eve calls", "Total night minutes", "Total night calls",
        "Total intl minutes", "Total intl calls", "Customer service calls"
    ]
    
    sample_data = {
        "Account length": [100.0, 150.0],
        "International plan": ["No", "Yes"],
        "Number vmail messages": [25.0, 0.0],
        "Total day minutes": [200.0, 150.0],
        "Total day calls": [100.0, 75.0],
        "Total eve minutes": [180.0, 220.0],
        "Total eve calls": [90.0, 80.0],
        "Total night minutes": [190.0, 160.0],
        "Total night calls": [85.0, 70.0],
        "Total intl minutes": [10.0, 12.0],
        "Total intl calls": [5.0, 6.0],
        "Customer service calls": [1.0, 4.0]
    }
    
    sample_df = pd.DataFrame(sample_data)
    for col_name in sample_df.columns:
        if col_name in ["International plan"]: # Categorical features
            sample_df[col_name] = sample_df[col_name].astype(str)
        else: # Numerical features
            sample_df[col_name] = sample_df[col_name].astype(float)

    predictions = mlflow_model.predict(sample_df)
    
    assert len(predictions) == 2
    assert all(p in [0.0, 1.0] for p in predictions), f"Unexpected prediction labels: {predictions}"

def test_data_cleaning_results(spark_session):
    """Tests if data cleaning results in expected non-null numerical columns."""
  
    test_raw_data = [
        ("StateA", "100", "415", "Phone1", "No", "Yes", "20", "150.5", "80", "25.0", "180.0", "90", "15.0", "200.0", "70", "10.0", "12.0", "5", "3.0", "1", "No"),
        ("StateB", "120", "NA", "Phone2", "Yes", "", "10", "", "70", "20.0", "200.0", "80", "18.0", "180.0", "60", "9.0", "10.0", "4", "2.0", "", "Yes"), # Area code as "NA", Number vmail messages empty, Customer service calls empty
        ("StateC", "", "400", "Phone3", "No", "No", "0", "100.0", "50", "10.0", "120.0", "60", "8.0", "150.0", "50", "7.0", "", "2", "1.5", "2", "No"), # Account length empty, Total intl minutes empty
    ]
    
    data_schema = StructType([
        StructField("State", StringType(), True),
        StructField("Account length", StringType(), True), 
        StructField("Area code", StringType(), True), 
        StructField("Phone", StringType(), True),
        StructField("International plan", StringType(), True),
        StructField("Voice mail plan", StringType(), True),
        StructField("Number vmail messages", StringType(), True), 
        StructField("Total day minutes", StringType(), True), 
        StructField("Total day calls", StringType(), True), 
        StructField("Total day charge", StringType(), True), 
        StructField("Total eve minutes", StringType(), True), 
        StructField("Total eve calls", StringType(), True), 
        StructField("Total eve charge", StringType(), True), 
        StructField("Total night minutes", StringType(), True), 
        StructField("Total night calls", StringType(), True), 
        StructField("Total night charge", StringType(), True), 
        StructField("Total intl minutes", StringType(), True), 
        StructField("Total intl calls", StringType(), True), 
        StructField("Total intl charge", StringType(), True), 
        StructField("Customer service calls", StringType(), True), 
        StructField("Churn", StringType(), True) 
    ])

    df_raw = spark_session.createDataFrame(test_raw_data, schema=data_schema)
    df_cleaned = clean_and_prepare_spark_df(df_raw)

    numerical_cols_to_check = [
        "Account length", "Area code", "Number vmail messages", "Total day minutes", 
        "Total day calls", "Total day charge", "Total eve minutes", "Total eve calls", 
        "Total eve charge", "Total night minutes", "Total night calls", 
        "Total night charge", "Total intl minutes", "Total intl calls", 
        "Total intl charge", "Customer service calls"
    ]
    
    null_counts = df_cleaned.select([sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in numerical_cols_to_check]).collect()[0].asDict()
    for col_name, count in null_counts.items():
        assert count == 0, f"Column '{col_name}' still contains nulls after cleaning."
    
    for f in df_cleaned.schema.fields:
        if f.name in numerical_cols_to_check:
            assert isinstance(f.dataType, DoubleType), f"Column '{f.name}' is not DoubleType, but {f.dataType}"


