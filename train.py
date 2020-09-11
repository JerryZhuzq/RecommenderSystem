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
def process_data():
    df = spark.read.csv('data/train.csv', header=True)
    df.write.parquet("data/train.parquet")
    df = spark.read.csv('data/test.csv', header=True)
    df.write.parquet("data/test.parquet")
    df = spark.read.csv('data/validation.csv', header=True)
    df.write.parquet("data/validation.parquet")
    pass

if __name__ == "__main__":
    # memory = f"{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g"
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
    
   
    als = ALS(rank=15, maxIter=5, regParam=0.001,userCol="user_id", itemCol="book_id", ratingCol="rating", seed=0, coldStartStrategy="drop")
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
    
    #crossval = CrossValidator(estimator=lr,estimatorParamMaps=paramGrid,evaluator=RankingMetrics(final.select('prediction', 'rating').rdd.map(tuple)),numFolds=5)
    #paramGrid = ParamGridBuilder().addGrid(als.rank, [5, 10, 20, 50, 100]).addGrid(als.regParam, [0.001, 0.01, 0.05, 0.1, 0.5]).build()
    #user_recs = model.recommendForAllUsers(5)
    #user_recs.show(10)
    # scoresAndLabels = spark.sparkContext.parallelize(df)
    metrics = RankingMetrics(final.select('prediction', 'rating').rdd.map(tuple))
    res = metrics.ndcgAt(500)
    Precision = metrics.precisionAt(500)
    #metrics.show(10)
    print(f"The NDCG evaluation result is: {res}")
    print(f"The PrecisionAtK evaluation result is: {Precision}")


    # df = spark.read.csv("data/train.csv", header=True)
    # df.write.parquet("data/train.parquet")
    # df = spark.read.csv("data/validation.csv")
    # df.write.parquet("data/validation.parquet")
