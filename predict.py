from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from utils import clean_transform_data
from datetime import datetime


def run_prediction(input_path: str, model_path: str, output_path: str):
    # Start SparkSession
    spark = SparkSession.builder \
         .appName("flight_price_pred") \
         .getOrCreate()

    # Load Model
    model = PipelineModel.load(model_path)

    # Load Input File
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Clean & Transform Data
    df_cleaned = clean_transform_data(df)

    # Make Predictions
    predictions = model.transform(df_cleaned)

    # Save Prediction Result
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{output_path}/prediction_result_{timestamp}"
    predictions.coalesce(1).select("legId", "prediction").write.csv(save_path, header=True, mode="overwrite")