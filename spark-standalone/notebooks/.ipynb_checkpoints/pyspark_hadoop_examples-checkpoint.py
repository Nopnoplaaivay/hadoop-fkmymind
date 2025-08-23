#!/usr/bin/env python3
"""
PySpark Examples with Hadoop Integration
Save this file as: notebooks/pyspark_hadoop_examples.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, max, min
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import time

# ========== 1. BASIC SPARK SESSION WITH HADOOP ==========
def create_spark_session():
    """Create SparkSession with Hadoop configuration"""
    spark = SparkSession.builder \
        .appName("PySpark-Hadoop-Examples") \
        .master("spark://spark-master:7077") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    
    print("Spark Session created successfully!")
    print(f"Spark Version: {spark.version}")
    print(f"Hadoop Version: {spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}")
    
    return spark

# ========== 2. READ/WRITE FILES FROM/TO HDFS ==========
def hdfs_file_operations(spark):
    """Demo reading and writing files to HDFS"""
    print("\n=== HDFS File Operations ===")
    
    # Create sample data
    data = [
        ("Alice", "Engineering", 75000),
        ("Bob", "Marketing", 65000),
        ("Charlie", "Engineering", 80000),
        ("Diana", "HR", 70000),
        ("Eve", "Marketing", 72000)
    ]
    
    columns = ["name", "department", "salary"]
    df = spark.createDataFrame(data, columns)
    
    # Write to HDFS
    hdfs_path = "hdfs://namenode:9000/spark-output/employees"
    df.write.mode("overwrite").parquet(hdfs_path)
    print(f"Data written to HDFS: {hdfs_path}")
    
    # Read from HDFS
    df_read = spark.read.parquet(hdfs_path)
    print("\nData read from HDFS:")
    df_read.show()
    
    # Write as CSV
    csv_path = "hdfs://namenode:9000/spark-output/employees.csv"
    df.write.mode("overwrite").option("header", "true").csv(csv_path)
    print(f"CSV written to: {csv_path}")
    
    return df

# ========== 3. WORDCOUNT EXAMPLE ==========
def wordcount_example(spark):
    """Classic WordCount with HDFS"""
    print("\n=== WordCount Example ===")
    
    # Create text file in HDFS
    text_data = spark.sparkContext.parallelize([
        "Apache Spark is a unified analytics engine",
        "Spark provides high-level APIs in Java Scala Python and R",
        "Spark runs on Hadoop YARN Kubernetes standalone or in the cloud",
        "Spark can access diverse data sources including HDFS Cassandra HBase and S3"
    ])
    
    # Save to HDFS
    text_path = "hdfs://namenode:9000/spark-input/text-data"
    text_data.saveAsTextFile(text_path)
    
    # Read and process
    text_df = spark.read.text(text_path)
    
    # WordCount using DataFrame API
    from pyspark.sql.functions import explode, split, lower
    
    words_df = text_df.select(
        explode(split(lower(col("value")), "\\s+")).alias("word")
    )
    
    word_counts = words_df.groupBy("word").count().orderBy(col("count").desc())
    
    print("Top 10 words:")
    word_counts.show(10)
    
    # Save results to HDFS
    output_path = "hdfs://namenode:9000/spark-output/wordcount"
    word_counts.write.mode("overwrite").json(output_path)
    print(f"WordCount results saved to: {output_path}")

# ========== 4. SQL OPERATIONS WITH HIVE TABLES ==========
def sql_operations(spark):
    """SQL operations with temporary views"""
    print("\n=== SQL Operations ===")
    
    # Create sample sales data
    sales_data = [
        ("2024-01-01", "Product-A", 100, 25.50),
        ("2024-01-01", "Product-B", 50, 35.00),
        ("2024-01-02", "Product-A", 120, 25.50),
        ("2024-01-02", "Product-C", 80, 45.00),
        ("2024-01-03", "Product-B", 70, 35.00),
    ]
    
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("product", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("price", FloatType(), True)
    ])
    
    sales_df = spark.createDataFrame(sales_data, schema)
    
    # Register as temp view
    sales_df.createOrReplaceTempView("sales")
    
    # SQL queries
    result1 = spark.sql("""
        SELECT product, 
               SUM(quantity) as total_quantity,
               SUM(quantity * price) as total_revenue
        FROM sales
        GROUP BY product
        ORDER BY total_revenue DESC
    """)
    
    print("Sales Summary by Product:")
    result1.show()
    
    # Daily revenue
    result2 = spark.sql("""
        SELECT date,
               COUNT(DISTINCT product) as products_sold,
               SUM(quantity * price) as daily_revenue
        FROM sales
        GROUP BY date
        ORDER BY date
    """)
    
    print("Daily Revenue:")
    result2.show()
    
    # Save to HDFS
    result1.write.mode("overwrite").parquet("hdfs://namenode:9000/spark-output/sales-summary")

# ========== 5. MACHINE LEARNING EXAMPLE ==========
def ml_example(spark):
    """Simple ML example with MLlib"""
    print("\n=== Machine Learning Example ===")
    
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Create sample data
    data = spark.createDataFrame([
        (1.0, 2.0, 3.0, 10.0),
        (2.0, 3.0, 4.0, 15.0),
        (3.0, 4.0, 5.0, 20.0),
        (4.0, 5.0, 6.0, 25.0),
        (5.0, 6.0, 7.0, 30.0),
        (6.0, 7.0, 8.0, 35.0),
    ], ["feature1", "feature2", "feature3", "label"])
    
    # Prepare features
    assembler = VectorAssembler(
        inputCols=["feature1", "feature2", "feature3"],
        outputCol="features"
    )
    
    data_assembled = assembler.transform(data)
    
    # Split data
    train_data, test_data = data_assembled.randomSplit([0.7, 0.3], seed=42)
    
    # Train model
    lr = LinearRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    print("Predictions:")
    predictions.select("features", "label", "prediction").show()
    
    # Evaluate model
    evaluator = RegressionEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="rmse"
    )
    
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Save model to HDFS
    model_path = "hdfs://namenode:9000/spark-models/linear-regression"
    model.write().overwrite().save(model_path)
    print(f"Model saved to: {model_path}")

# ========== 6. STREAMING EXAMPLE (STRUCTURED STREAMING) ==========
def streaming_example(spark):
    """Structured Streaming example"""
    print("\n=== Structured Streaming Example ===")
    
    # Create streaming DataFrame from HDFS directory
    streaming_df = spark.readStream \
        .option("maxFilesPerTrigger", 1) \
        .text("hdfs://namenode:9000/streaming-input/")
    
    # Word count on streaming data
    from pyspark.sql.functions import explode, split
    
    words = streaming_df.select(
        explode(split(streaming_df.value, " ")).alias("word")
    )
    
    word_counts = words.groupBy("word").count()
    
    # Write stream to console (for demo)
    query = word_counts.writeStream \
        .outputMode("complete") \
        .format("console") \
        .trigger(processingTime='10 seconds') \
        .start()
    
    print("Streaming query started. Waiting for data...")
    # Run for 30 seconds then stop
    time.sleep(30)
    query.stop()
    print("Streaming stopped.")

# ========== 7. PERFORMANCE OPTIMIZATION EXAMPLE ==========
def performance_optimization(spark):
    """Demonstrate performance optimization techniques"""
    print("\n=== Performance Optimization ===")
    
    # Create large dataset
    large_df = spark.range(0, 10000000).toDF("id")
    
    # Add some computed columns
    from pyspark.sql.functions import rand, when
    
    df_with_cols = large_df \
        .withColumn("random_value", rand()) \
        .withColumn("category", when(col("id") % 3 == 0, "A")
                                .when(col("id") % 3 == 1, "B")
                                .otherwise("C"))
    
    # Cache DataFrame for reuse
    df_with_cols.cache()
    
    # Perform multiple operations
    print("Count by category (with caching):")
    start_time = time.time()
    df_with_cols.groupBy("category").count().show()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Second operation uses cached data
    print("\nAverage random value by category (using cache):")
    start_time = time.time()
    df_with_cols.groupBy("category").agg(avg("random_value")).show()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Repartition for better parallelism
    df_repartitioned = df_with_cols.repartition(10, "category")
    
    # Save with partitioning
    output_path = "hdfs://namenode:9000/spark-output/partitioned-data"
    df_repartitioned.write \
        .mode("overwrite") \
        .partitionBy("category") \
        .parquet(output_path)
    print(f"Data saved with partitioning to: {output_path}")

# ========== 8. JOIN OPERATIONS ==========
def join_operations(spark):
    """Demonstrate various join operations"""
    print("\n=== Join Operations ===")
    
    # Create employee data
    employees = spark.createDataFrame([
        (1, "Alice", "Engineering"),
        (2, "Bob", "Marketing"),
        (3, "Charlie", "Engineering"),
        (4, "Diana", "HR"),
        (5, "Eve", "Marketing")
    ], ["emp_id", "name", "department"])
    
    # Create salary data
    salaries = spark.createDataFrame([
        (1, 75000),
        (2, 65000),
        (3, 80000),
        (4, 70000),
        (6, 60000)  # Note: emp_id 6 doesn't exist in employees
    ], ["emp_id", "salary"])
    
    # Inner join
    print("Inner Join Result:")
    inner_join = employees.join(salaries, on="emp_id", how="inner")
    inner_join.show()
    
    # Left join
    print("Left Join Result:")
    left_join = employees.join(salaries, on="emp_id", how="left")
    left_join.show()
    
    # Broadcast join for small tables
    from pyspark.sql.functions import broadcast
    
    departments = spark.createDataFrame([
        ("Engineering", "Tech"),
        ("Marketing", "Business"),
        ("HR", "Operations")
    ], ["department", "division"])
    
    print("Broadcast Join Result:")
    broadcast_join = employees.join(broadcast(departments), on="department")
    broadcast_join.show()
    
    # Save results
    inner_join.write.mode("overwrite").json("hdfs://namenode:9000/spark-output/join-results")

# ========== 9. WINDOW FUNCTIONS ==========
def window_functions(spark):
    """Demonstrate window functions"""
    print("\n=== Window Functions ===")
    
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead
    
    # Create sales data
    sales = spark.createDataFrame([
        ("Alice", "2024-01", 5000),
        ("Bob", "2024-01", 4500),
        ("Charlie", "2024-01", 6000),
        ("Alice", "2024-02", 5500),
        ("Bob", "2024-02", 4800),
        ("Charlie", "2024-02", 5800),
        ("Alice", "2024-03", 5200),
        ("Bob", "2024-03", 5000),
        ("Charlie", "2024-03", 6200)
    ], ["employee", "month", "sales_amount"])
    
    # Define window specifications
    window_by_month = Window.partitionBy("month").orderBy(col("sales_amount").desc())
    window_by_employee = Window.partitionBy("employee").orderBy("month")
    
    # Apply window functions
    result = sales \
        .withColumn("rank_in_month", rank().over(window_by_month)) \
        .withColumn("dense_rank_in_month", dense_rank().over(window_by_month)) \
        .withColumn("row_number", row_number().over(window_by_month)) \
        .withColumn("prev_month_sales", lag("sales_amount", 1).over(window_by_employee)) \
        .withColumn("next_month_sales", lead("sales_amount", 1).over(window_by_employee))
    
    print("Sales with Window Functions:")
    result.orderBy("month", "rank_in_month").show()
    
    # Save to HDFS
    result.write.mode("overwrite").parquet("hdfs://namenode:9000/spark-output/window-functions")

# ========== 10. MAIN EXECUTION ==========
def main():
    """Main function to run all examples"""
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Run examples
        df = hdfs_file_operations(spark)
        wordcount_example(spark)
        sql_operations(spark)
        ml_example(spark)
        # streaming_example(spark)  # Uncomment if you have streaming data
        performance_optimization(spark)
        join_operations(spark)
        window_functions(spark)
        
        print("\n=== All Examples Completed Successfully! ===")
        print("\nCheck HDFS for output files:")
        print("hdfs dfs -ls /spark-output/")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Stop Spark session
        spark.stop()
        print("Spark session closed.")

if __name__ == "__main__":
    main()