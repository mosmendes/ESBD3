# app.py

import streamlit as st
import os
import mlflow.pyfunc
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import mlflow.tracking

# --- Configuração da página Streamlit ---
st.set_page_config(page_title="Previsão de Churn", layout="centered")
st.title("Sistema de Previsão de Churn de Clientes")

# --- Campos de entrada para o usuário ---
st.subheader("Insira os dados do cliente para prever o Churn:")

input_account_length = st.number_input("Duração da Conta (Account length)", min_value=0, value=100)
input_international_plan = st.selectbox("Plano Internacional (International plan)", ["Yes", "No"])
input_number_vmail_messages = st.number_input("Número de Mensagens de Voz (Number vmail messages)", min_value=0, value=0)
input_total_day_minutes = st.number_input("Total de Minutos Diurnos (Total day minutes)", min_value=0.0, value=100.0, format="%.2f")
input_total_day_calls = st.number_input("Total de Chamadas Diurnas (Total day calls)", min_value=0, value=100)
input_total_eve_minutes = st.number_input("Total de Minutos Noturnos (Total eve minutes)", min_value=0.0, value=100.0, format="%.2f")
input_total_eve_calls = st.number_input("Total de Chamadas Noturnas (Total eve calls)", min_value=0, value=100)
input_total_night_minutes = st.number_input("Total de Minutos da Madrugada (Total night minutes)", min_value=0.0, value=100.0, format="%.2f")
input_total_night_calls = st.number_input("Total de Chamadas da Madrugada (Total night calls)", min_value=0, value=100)
input_total_intl_minutes = st.number_input("Total de Minutos Internacionais (Total intl minutes)", min_value=0.0, value=10.0, format="%.2f")
input_total_intl_calls = st.number_input("Total de Chamadas Internacionais (Total intl calls)", min_value=0, value=5)
input_customer_service_calls = st.number_input("Chamadas para Atendimento ao Cliente (Customer service calls)", min_value=0, value=1)


# --- Funções para carregar SparkSession e Modelo MLflow (com cache) ---
@st.cache_resource
def get_spark_session():
    spark_tmp_dir = os.path.join(os.getcwd(), "spark_streamlit_tmp")
    os.makedirs(spark_tmp_dir, exist_ok=True)
    return SparkSession.builder \
        .appName("StreamlitSparkModelLoader") \
        .config("spark.local.dir", spark_tmp_dir) \
        .config("spark.sql.warehouse.dir", os.path.join(spark_tmp_dir, "spark-warehouse-streamlit")) \
        .getOrCreate()

spark = get_spark_session()

@st.cache_resource
def load_mlflow_churn_model():
    try:
        tracking_uri = "sqlite:///C:/MONICA/Estudos/mlf_data/mlflow.db" 

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

                model_uri = latest_run.info.artifact_uri + "/churn_model"

                st.info(f"Carregando modelo da URI: {model_uri}")
                return mlflow.pyfunc.load_model(model_uri)
            else:
                st.error("Nenhuma execução (run) encontrada para o experimento 'churn_prediction'.")
                return None
        else:
            st.error("Experimento 'churn_prediction' não encontrado no MLflow Tracking Server.")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo MLflow: {e}. Certifique-se de que o caminho do MLflow Tracking URI está correto e o MLflow está acessível.")
        return None


model_mlflow = load_mlflow_churn_model()

# --- Lógica de Previsão ---
if st.button("Prever Churn"):
    if model_mlflow:
        if all(val is not None for val in [
            input_account_length, input_international_plan, input_number_vmail_messages,
            input_total_day_minutes, input_total_day_calls, input_total_eve_minutes,
            input_total_eve_calls, input_total_night_minutes, input_total_night_calls,
            input_total_intl_minutes, input_total_intl_calls, input_customer_service_calls
        ]):
            input_data = [(
                "No", 
                input_account_length,
                input_international_plan,
                input_number_vmail_messages,
                input_total_day_minutes,
                input_total_day_calls,
                input_total_eve_minutes,
                input_total_eve_calls,
                input_total_night_minutes,
                input_total_night_calls,
                input_total_intl_minutes,
                input_total_intl_calls,
                input_customer_service_calls
            )]

            input_schema = StructType([
                StructField("Churn", StringType(), True),
                StructField("Account length", IntegerType(), True),
                StructField("International plan", StringType(), True),
                StructField("Number vmail messages", IntegerType(), True),
                StructField("Total day minutes", DoubleType(), True),
                StructField("Total day calls", IntegerType(), True),
                StructField("Total eve minutes", DoubleType(), True),
                StructField("Total eve calls", IntegerType(), True),
                StructField("Total night minutes", DoubleType(), True),
                StructField("Total night calls", IntegerType(), True),
                StructField("Total intl minutes", DoubleType(), True),
                StructField("Total intl calls", IntegerType(), True),
                StructField("Customer service calls", IntegerType(), True)
            ])
            
            input_df_spark = spark.createDataFrame(input_data, schema=input_schema)
            input_pandas_df = input_df_spark.toPandas()
            
            predictions = model_mlflow.predict(input_pandas_df)
            
            predicted_label_index = predictions[0] 
            predicted_churn = "Sim (Churn)" if predicted_label_index == 1.0 else "Não (Sem Churn)"
            
            st.success(f"Previsão de Churn: **{predicted_churn}**")
        else:
            st.warning("Por favor, preencha todos os campos para realizar a previsão.")
    else:
        st.error("O modelo de previsão de Churn não pôde ser carregado. Verifique os erros acima.")
