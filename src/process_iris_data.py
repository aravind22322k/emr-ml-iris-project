import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark Session
spark = SparkSession.builder.appName("IrisProcessing").getOrCreate()

# GitHub raw link for the Iris dataset
github_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/data/iris_data.csv"

# Download the dataset locally
local_file = "/tmp/iris_data.csv"
os.system(f"curl -o {local_file} {github_url}")

# Load dataset into Spark DataFrame
df = spark.read.csv(local_file, header=True, inferSchema=True)

# Perform preprocessing (e.g., feature scaling)
processed_df = df.withColumn("scaled_feature1", col("feature1") / 10.0)

# Save processed data locally as a Parquet file
processed_output = "/tmp/processed_iris"
processed_df.write.parquet(processed_output, mode="overwrite")

print(f"Processed data saved at {processed_output}")

# Stop Spark Session
spark.stop()
