import math
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
from sklearn import preprocessing
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from scipy.sparse import coo_matrix
import pandas as pd
import time


def trans(train_df,test_df):
    id_cols = ['user_id', 'book_id']
    trans_cat_train = dict()
    trans_cat_test = dict()
    test_df = test_df[(test_df['user_id'].isin(train_df['user_id'])) & (test_df['book_id'].isin(train_df['book_id']))]
    for k in id_cols:
        cate_enc = preprocessing.LabelEncoder()
        trans_cat_train[k] = cate_enc.fit_transform(train_df[k].values)
        trans_cat_test[k] = cate_enc.transform(test_df[k].values)
    cate_enc = preprocessing.LabelEncoder()
    ratings = dict()
    ratings['train'] = cate_enc.fit_transform(train_df.rating)
    ratings['test'] = cate_enc.transform(test_df.rating)
    n_users = len(np.unique(trans_cat_train['user_id']))
    n_items = len(np.unique(trans_cat_train['book_id']))
    train = coo_matrix((ratings['train'], (trans_cat_train['user_id'], trans_cat_train['book_id'])),shape=(n_users, n_items))
    test = coo_matrix((ratings['test'], (trans_cat_test['user_id'], trans_cat_test['book_id'])),shape=(n_users, n_items))
    return train, test


if __name__ == "__main__":
    spark = (SparkSession.builder
             .master("local")
             .appName("TandonProud_als")
             .config("spark.executor.memory", "16g")
             .config("spark.driver.memory", "16g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    train = spark.read.csv(f'{sys.argv[1]}/train.csv',header=True).sample(fraction=1,seed=0).toPandas().drop(['is_read','is_reviewed'],axis=1)
    test = spark.read.csv(f'{sys.argv[1]}/test.csv',header=True).sample(fraction=1,seed=0).toPandas().drop(['is_read','is_reviewed'],axis=1)
    validation = spark.read.csv(f'{sys.argv[1]}/validation.csv',header=True).sample(fraction=1,seed=0).toPandas().drop(['is_read','is_reviewed'],axis=1)

    train, test = trans(train,validation)
    model = LightFM(no_components=5,loss='warp')
    StartT = time.time()
    model.fit(train,epochs=30,num_threads=5)
    EndT = time.time()
    t = EndT - StartT
    print(f"The running time is : {t}")
    PrecisionAtK = precision_at_k(model, test, k=500).mean()
    
    print(f"The Precision At K evaluation result is: {PrecisionAtK}")

