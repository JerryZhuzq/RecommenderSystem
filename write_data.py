# import psutil
import sys
import math
import pyspark
from pyspark.sql import SparkSession


if __name__ == '__main__':
    # memory = f"{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g"
    spark = (SparkSession.builder
             .master("local")
             .appName("TandonProud_als")
             # .config("spark.executor.memory", memory)
             # .config("spark.driver.memory", memory)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(f'{sys.argv[1]}/train.csv', header=True, inferSchema=True)
    df.write.parquet(f'{sys.argv[1]}/train.parquet')
    df = spark.read.csv(f'{sys.argv[1]}/test.csv', header=True, inferSchema=True)
    df.write.parquet(f'{sys.argv[1]}/test.parquet')
    df = spark.read.csv(f'{sys.argv[1]}/validation.csv', header=True, inferSchema=True)
    df.write.parquet(f'{sys.argv[1]}/validation.parquet')
