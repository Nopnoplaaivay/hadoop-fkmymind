from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("TestSpark") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

print("Spark Version:", spark.version)

# Create simple DataFrame
data = [("Hello", 1), ("World", 2), ("Spark", 3)]
df = spark.createDataFrame(data, ["word", "count"])

print("\nDataFrame content:")
df.show()

# Test HDFS write
try:
    df.write.mode("overwrite").parquet("hdfs://namenode:9000/test-output")
    print("✓ Successfully wrote to HDFS")
except Exception as e:
    print(f"✗ HDFS write failed: {e}")

spark.stop()