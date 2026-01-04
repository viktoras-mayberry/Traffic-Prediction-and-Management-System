"""
Apache Spark Streaming Pipeline
Real-time data processing and aggregation
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.streaming import StreamingQuery
from typing import Optional, Dict, List
import yaml
import pandas as pd
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SparkStreamingPipeline:
    """
    Apache Spark streaming pipeline for real-time traffic data processing.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        app_name: str = "TrafficPredictionSystem"
    ):
        """
        Initialize Spark streaming pipeline.
        
        Args:
            config_path: Path to config.yaml file
            app_name: Spark application name
        """
        self.config_path = config_path
        self.app_name = app_name
        self.spark = None
        self.config = self._load_config()
        self._initialize_spark()
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        default_config = {
            "master": "local[*]",
            "config": {
                "spark.sql.shuffle.partitions": "200",
                "spark.streaming.batchDuration": "10",
                "spark.sql.streaming.checkpointLocation": "data/checkpoints",
                "spark.driver.memory": "4g",
                "spark.executor.memory": "4g"
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if 'spark' in file_config:
                        default_config.update(file_config['spark'])
            except Exception as e:
                logger.warning(f"Could not load Spark config: {e}")
        
        return default_config
    
    def _initialize_spark(self):
        """Initialize Spark session."""
        try:
            builder = SparkSession.builder.appName(self.app_name)
            
            # Set master
            builder = builder.master(self.config.get("master", "local[*]"))
            
            # Set configuration
            for key, value in self.config.get("config", {}).items():
                builder = builder.config(key, value)
            
            self.spark = builder.getOrCreate()
            logger.info("Spark session initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Spark: {e}")
            raise
    
    def create_streaming_source(
        self,
        source_type: str = "kafka",
        options: Optional[Dict] = None
    ):
        """
        Create a streaming data source.
        
        Args:
            source_type: Type of source (kafka, socket, file)
            options: Source-specific options
        
        Returns:
            Streaming DataFrame
        """
        if options is None:
            options = {}
        
        try:
            if source_type == "kafka":
                return self.spark.readStream \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", options.get("bootstrap_servers", "localhost:9092")) \
                    .option("subscribe", options.get("topic", "traffic_data")) \
                    .load()
            
            elif source_type == "socket":
                return self.spark.readStream \
                    .format("socket") \
                    .option("host", options.get("host", "localhost")) \
                    .option("port", options.get("port", 9999)) \
                    .load()
            
            elif source_type == "file":
                return self.spark.readStream \
                    .format("csv") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .schema(self._get_traffic_schema()) \
                    .load(options.get("path", "data/raw/streaming"))
            
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            logger.error(f"Error creating streaming source: {e}")
            raise
    
    def process_traffic_stream(
        self,
        stream_df,
        window_duration: str = "5 minutes",
        slide_duration: str = "1 minute"
    ):
        """
        Process traffic data stream with windowed aggregations.
        
        Args:
            stream_df: Input streaming DataFrame
            window_duration: Window duration (e.g., "5 minutes")
            slide_duration: Slide duration (e.g., "1 minute")
        
        Returns:
            Processed streaming DataFrame
        """
        try:
            # Parse JSON if needed
            if "value" in stream_df.columns:
                from pyspark.sql.functions import from_json
                schema = self._get_traffic_schema()
                stream_df = stream_df.select(
                    from_json(col("value").cast("string"), schema).alias("data")
                ).select("data.*")
            
            # Ensure timestamp column exists
            if "timestamp" not in stream_df.columns:
                stream_df = stream_df.withColumn("timestamp", current_timestamp())
            else:
                stream_df = stream_df.withColumn("timestamp", to_timestamp(col("timestamp")))
            
            # Windowed aggregations
            windowed = stream_df \
                .withWatermark("timestamp", "10 minutes") \
                .groupBy(
                    window(col("timestamp"), window_duration, slide_duration),
                    col("sensor_id").alias("location_id")
                ) \
                .agg(
                    avg("speed").alias("avg_speed"),
                    sum("vehicle_count").alias("total_vehicles"),
                    avg("occupancy").alias("avg_occupancy"),
                    count("*").alias("record_count")
                ) \
                .select(
                    col("window.start").alias("window_start"),
                    col("window.end").alias("window_end"),
                    col("location_id"),
                    col("avg_speed"),
                    col("total_vehicles"),
                    col("avg_occupancy"),
                    col("record_count")
                )
            
            logger.info("Created windowed aggregations")
            return windowed
            
        except Exception as e:
            logger.error(f"Error processing traffic stream: {e}")
            raise
    
    def write_streaming_sink(
        self,
        stream_df,
        sink_type: str = "console",
        options: Optional[Dict] = None,
        checkpoint_location: Optional[str] = None
    ) -> StreamingQuery:
        """
        Write streaming data to a sink.
        
        Args:
            stream_df: Streaming DataFrame to write
            sink_type: Type of sink (console, kafka, parquet, memory)
            options: Sink-specific options
            checkpoint_location: Checkpoint location for fault tolerance
        
        Returns:
            StreamingQuery object
        """
        if options is None:
            options = {}
        
        if checkpoint_location is None:
            checkpoint_location = self.config.get("config", {}).get(
                "spark.sql.streaming.checkpointLocation",
                "data/checkpoints"
            )
        
        try:
            writer = stream_df.writeStream \
                .outputMode("update") \
                .option("checkpointLocation", checkpoint_location)
            
            if sink_type == "console":
                query = writer.format("console").start()
            
            elif sink_type == "kafka":
                query = writer \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", options.get("bootstrap_servers", "localhost:9092")) \
                    .option("topic", options.get("topic", "processed_traffic")) \
                    .start()
            
            elif sink_type == "parquet":
                query = writer \
                    .format("parquet") \
                    .option("path", options.get("path", "data/processed/streaming")) \
                    .start()
            
            elif sink_type == "memory":
                query = writer \
                    .format("memory") \
                    .queryName(options.get("table_name", "traffic_stream")) \
                    .start()
            
            else:
                raise ValueError(f"Unsupported sink type: {sink_type}")
            
            logger.info(f"Started streaming query to {sink_type} sink")
            return query
            
        except Exception as e:
            logger.error(f"Error writing streaming sink: {e}")
            raise
    
    def _get_traffic_schema(self) -> StructType:
        """Get schema for traffic data."""
        return StructType([
            StructField("sensor_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("speed", DoubleType(), True),
            StructField("vehicle_count", IntegerType(), True),
            StructField("occupancy", DoubleType(), True),
            StructField("flow_rate", DoubleType(), True)
        ])
    
    def stop(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")
    
    def process_batch_data(self, df, output_path: str = "data/processed/batch"):
        """
        Process batch data (non-streaming).
        
        Args:
            df: Input DataFrame
            output_path: Output path for processed data
        
        Returns:
            Processed DataFrame
        """
        try:
            # Convert pandas DataFrame to Spark DataFrame if needed
            if isinstance(df, type(pd.DataFrame())):
                spark_df = self.spark.createDataFrame(df)
            else:
                spark_df = df
            
            # Apply transformations
            processed = spark_df \
                .withColumn("timestamp", to_timestamp(col("timestamp"))) \
                .withColumn("hour", hour(col("timestamp"))) \
                .withColumn("day_of_week", dayofweek(col("timestamp"))) \
                .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
            
            # Save if output path provided
            if output_path:
                processed.write.mode("overwrite").parquet(output_path)
                logger.info(f"Saved processed batch data to {output_path}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing batch data: {e}")
            raise

