from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, trim
from pyspark.sql.functions import to_date, year, month, datediff, from_unixtime, hour

def clean_transform_data(df: DataFrame) -> DataFrame:

    # Clean & Transform Numeric Columns
    numeric_columns = ["totalFare", "seatsRemaining", "totalTravelDistance"]
    df1_cleaned = df
    for column in numeric_columns:
        df1_cleaned = df1_cleaned.filter(trim(col(column)).rlike("^\d+(\.\d+)?$"))
        df1_cleaned = df1_cleaned.withColumn(column, col(column).cast(IntegerType()))

    # Clean & Transform Date columns
    date_columns = ["searchDate", "flightDate"]
    for column in date_columns:
        df1_cleaned = df1_cleaned.withColumn(f"{column}_dt", to_date(column, "yyyy-MM-dd")) \
            .withColumn(f"{column}_year", year(f"{column}_dt")) \
            .withColumn(f"{column}_month", month(f"{column}_dt"))

    df1_cleaned = df1_cleaned.withColumn("daysUntilFlight", datediff("flightDate_dt", "searchDate_dt"))

    # Clean & Transform Time columns
    time_columns = ["segmentsDepartureTimeEpochSeconds", "segmentsArrivalTimeEpochSeconds"]
    for column in time_columns:
        df1_cleaned = df1_cleaned.withColumn(f"{column}_ts", from_unixtime(column)) \
            .withColumn(f"{column}_hr", hour(f"{column}_ts"))

    # Clean & Transform Boolean columns
    bool_columns = ["isBasicEconomy", "isRefundable", "isNonStop"]
    for column in bool_columns:
        df1_cleaned = df1_cleaned.withColumn(f"{column}_dg", col(column).cast("int"))

    return df1_cleaned