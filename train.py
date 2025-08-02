from pyspark.sql import SparkSession
from datetime import datetime
from utils import clean_transform_data
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor


def train_model(input_path: str, model_path: str):
    # Start SparkSession
    spark = SparkSession.builder \
        .appName("flight_analysis") \
        .master("local[*]") \
        .getOrCreate()

    # Load Data
    df1 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input_path)

    # Extract a smaller dataset
    small_df1 = df1.sample(True, 0.005, seed=42)

    # Clean & Transform Data
    small_df1_cleaned = clean_transform_data(small_df1)


    # Build, Test & Evaluate Model
    # Define categorical and numeric columns
    categorical_cols = ["startingAirport", "destinationAirport", "segmentsAirlineCode",
                        "segmentsDepartureTimeEpochSeconds_hr", "flightDate_year"]
    numeric_cols = ["seatsRemaining", "totalTravelDistance", "seatsRemaining", "daysUntilFlight", "isBasicEconomy_dg",
                    "isRefundable_dg", "isNonStop_dg"]
    label_col = "totalFare"

    # Train/test split
    train_df, test_df = small_df1_cleaned.randomSplit([0.8, 0.2], seed=45)


    # Train Linear Regression model
    # Step 1: Index and one-hot encode categorical columns
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec") for col in categorical_cols]

    # Step 2: Assemble features into a single vector column
    assembler_inputs = [f"{col}_vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Step 3: Train Model
    lr = LinearRegression(featuresCol="features", labelCol=label_col)

    # Step 4: Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler] + [lr])
    lr_model = pipeline.fit(train_df)

    # Step 5: Save Model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{model_path}/linear_regression_{timestamp}"
    lr_model.write().overwrite().save(save_path)

    # Decision Tree Regression
    # Step 1: Index and one-hot encode categorical columns
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in categorical_cols]

    # Step 2: Assemble features into a single vector column
    assembler_inputs = [f"{col}_idx" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Step 3: Train Model
    dt = DecisionTreeRegressor(featuresCol="features", labelCol=label_col, maxBins=360)

    # Step 3: Pipeline
    pipeline = Pipeline(stages=indexers + [assembler] + [dt])
    dt_model = pipeline.fit(train_df)

    # Step 5: Save Model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{model_path}/decision_tree_regression_{timestamp}"
    dt_model.write().overwrite().save(save_path)
