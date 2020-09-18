# import psutil
import math
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics,RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time


if __name__ == "__main__":
    spark = (SparkSession.builder
             .master("local")
             .appName("TandonProud_als")
             .config("spark.executor.memory", "8g")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    train = spark.read.parquet(f'{sys.argv[1]}/train.parquet')
    test = spark.read.parquet(f'{sys.argv[1]}/test.parquet')
    validation = spark.read.parquet(f'{sys.argv[1]}/validation.parquet')

    als = ALS(rank=15, maxIter=5, regParam=0.001,userCol="user_id", itemCol="book_id", ratingCol="rating", seed=0,
              coldStartStrategy="drop")
    StartT = time.time()
    model = als.fit(train)
    EndT = time.time()
    T = EndT - StartT
    print(f"The running time is: {T}")
    predictions = model.transform(test)

    predictions = predictions.orderBy(predictions.prediction.desc())
    final_prediction = predictions.filter(predictions.prediction >= 0).groupBy("user_id").agg(F.collect_list("book_id").alias("prediction"))
    predictions = predictions.orderBy(predictions.rating.desc())
    final_rating = predictions.groupBy("user_id").agg(F.collect_list("book_id").alias("rating"))
    final = final_prediction.join(final_rating, final_prediction.user_id == final_rating.user_id, 'inner')\
        .select(final_prediction.user_id, final_prediction.prediction, final_rating.rating)
    metrics = RankingMetrics(final.select('prediction', 'rating').rdd.map(tuple))
    res = metrics.ndcgAt(500)
    Precision = metrics.precisionAt(500)
    print(f"The NDCG evaluation result is: {res}")
    print(f"The PrecisionAtK evaluation result is: {Precision}")

