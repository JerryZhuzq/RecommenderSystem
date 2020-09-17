# Recommender System
##### Physical objects occupy space

- Brick-and-mortar shops must satisfy physical constraints
- Curators must prioritize for some notion of utility
- Serving the most customers, maximizing sales/profit, etc.

##### This is not true for digital items!

- The web, e-books, news article, movies, music, ... take up no physical space
- Without curation, this quickly becomes overwhelming

## The data set

In this project, we'll use the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by

> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.

## Data splitting and Train using ALS

Relevant code script: csv_split.py/write_data.py/train.py

For data splitting and subsampling, the job is to split a whole data set into three parts including training set, testing set and validation set. The requirement is that both of testing set and validation set should contain 20% of total users, at the same time, all the users in these two sets should also exist in training set since the algorithm cannot predict users not appeared in training set. Therefore, we obtain the final format of data: training set has 60% ~ 80% of original data with 100% user id, while the other two data sets have 10% ~ 20% of original data with 20% user id each. In code, it just needs two functions to achieve the data splitting and subsampling.

```python
def count_interactions(filehandler, delimiter=','):
    reader = csv.reader(open(filehandler,'r'), delimiter=delimiter)
    store = {}
    for i, row in enumerate(reader):
        store[row[0]] = store.get(row[0], 0) + 1

    new_store = {k: v for k, v in list(store.items()) if v > 10}
    return new_store
```

```python
def split(filehandler, outputfile, delimiter=','):
    store = count_interactions(filehandler, delimiter)
    train_store = {}
    test_store = {}
    val_store = {}

    reader = csv.reader(open(filehandler,'r'), delimiter=delimiter)
    header = next(reader)
    train_writer = csv.writer(open(f'{outputfile}/train_s.csv', 'w')).sample(frac=0.5, random_state=1)
    test_writer = csv.writer(open(f'{outputfile}/test_s.csv', 'w')).sample(frac=0.5, random_state=1)
    val_writer = csv.writer(open(f'{outputfile}/validation_s.csv', 'w')).sample(frac=0.5, random_state=1)
    train_writer.writerow(header)
    test_writer.writerow(header)
    val_writer.writerow(header)
    for i, row in enumerate(reader):
        if row[0] in store:
            if row[0] in train_store or len(train_store) <= 0.6*len(store):
                train_store[row[0]] = train_store.get(row[0],0) + 1
                train_writer.writerow(row)
            elif row[0] in test_store or len(test_store) <= 0.2*len(store):
                test_store[row[0]] = test_store.get(row[0], 0) + 1
                if test_store.get(row[0], 0) <= store[row[0]]/2:
                    test_writer.writerow(row)
                else:
                    train_writer.writerow(row)
            else:
                val_store[row[0]] = val_store.get(row[0], 0) + 1
                if val_store.get(row[0], 0) <= store[row[0]]/2:
                    val_writer.writerow(row)
                else:
                    train_writer.writerow(row)
```

In addition, the count_interactions function filters out user_id with less than 10 items. The split function splits and subsamples the input csv file, and generates training, testing and validation csv file. Then data splitting and subsampling phase is done. The write_data.py code is write csv data into parquet format for more efficient access. After using ALS for training and predicting, we have a ground_truth column and a prediction column. I choose to aggregate each row of the same user_id into a single row with two columns, one for prediction and one for ground_truth. The format is like this:

| User_id | Prediction Rating             | Ground_truth Rating           |
| ------- | ----------------------------- | ----------------------------- |
| 001     | [book_id 001, 004, 002, ....] | [book_id 001, 002, 003, ....] |
| 002     | [book_id 015, 034, 031, ....] | [book_id 034, 031, 016, ....] |
| ...     | ...                           | ...                           |

The cell of book_id means the rank of rating in descending order. I filtered out the books whose prediction rating were less than zero. From my point of view, is because the prediction rating less than 0 indicating that this book would not be recommended for this user.

```python
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
    
    metrics = RankingMetrics(final.select('prediction', 'rating').rdd.map(tuple))
    res = metrics.ndcgAt(500)
    Precision = metrics.precisionAt(500)
    print(f"The NDCG evaluation result is: {res}")
    print(f"The PrecisionAtK evaluation result is: {Precision}")

```

## Evaluation

Relevant code script: train.py

We choose Normalized Discounted Cumulative Gain and Precision at k as our ranking metric with K = 500 since we should be based on predicted top 500 items for each user. And we did hyper parameter tuning of the rank of the latent factors and the regularization parameter to optimize performance and running time of model fitting.

Scores for validation and test data are pretty much the same in our evaluation. Following is the validation data evaluation result: 

NDCG:

| Rank/regParam | regParam = 0.001 | regParam = 0.05 | regParam = 1 |
| ------------- | ---------------- | --------------- | ------------ |
| Rank = 5      | 95.15%           | 96.94%          | 98.88%       |
| Rank = 15     | 90.17%           | 95.31%          | 97.23%       |
| Rank = 30     | 83.24%           | 92.64%          | 94.16%       |

Precision at K:

| Rank/regParam | regParam = 0.001 | regParam = 0.05 | regParam = 1 |
| ------------- | ---------------- | --------------- | ------------ |
| Rank = 5      | 4.64%            | 4.73%           | 5.03%        |
| Rank = 15     | 4.26%            | 4.60%           | 4.93%        |
| Rank = 30     | 3.78%            | 4.19%           | 4.57%        |

Running time:

| Rank/regParam | regParam = 0.001 | regParam = 0.05 | regParam = 1 |
| ------------- | ---------------- | --------------- | ------------ |
| Rank = 5      | 1231s            | 861s            | 1141s        |
| Rank = 15     | 1782s            | 1231s           | 1850s        |
| Rank = 30     | 2635s            | 1769s           | 2459s        |

Since we use model.transform to make predictions, NDCG is relatively much higher than the precision at K.

## Extension

Relevant code script: extension.py

We choose to compare Spark's parallel ALS model to a LightFM model as our project extension.

```python
def Trans(train_df,test_df):
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

    train, test = Trans(train,validation)
    model=LightFM(no_components=5,loss='warp')
    StartT = time.time()
    model.fit(train,epochs=30,num_threads=5)
    EndT = time.time()
    t = EndT - StartT
    print(f"The running time is : {t}")
    PrecisionAtK = precision_at_k(model, test, k=500).mean()   
    print(f"The Precision At K evaluation result is: {PrecisionAtK}")
```

In Trans function, we transfer train data to sparse matrix form using sklearn preprocessing and scipy coo_matrix function to fit the Lightfm model. Also, we choose Weighted Approximate-Rank Pairwise as loss function since it is desired of precision at k metrics. LightFM model fitting time of full dataset with rank = 5 and regParam = 0.05 is 15263 seconds, which is much longer than ALS model(861 seconds). The precision at K = 500 is 3.57%, which is also lower than ALS model output.

LightFM fitting time with different size of data:

| Model/dataset size | Lightfm | ALS  |
| ------------------ | ------- | ---- |
| 10%                | 3881s   | 536s |
| 100%               | 15263s  | 861s |

Contribution of each member:

Zhongqi Zhu: Data splitting, Training using ALS, Evaluation

Henglin Li: Evaluation, Hyper-Tuning to optimize evaluation result

