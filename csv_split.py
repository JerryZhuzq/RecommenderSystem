import os
import csv
import sys
import pandas as pd
import pyspark
from pyspark.sql import SparkSession


def count_interactions(filehandler, delimiter=','):
    reader = csv.reader(open(filehandler,'r'), delimiter=delimiter)
    store = {}
    for i, row in enumerate(reader):
        store[row[0]] = store.get(row[0], 0) + 1

    new_store = {k: v for k, v in list(store.items()) if v > 10}
    return new_store


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


if __name__ == '__main__':
    if len(sys.argv) > 0:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        split(input_file, output_file)



