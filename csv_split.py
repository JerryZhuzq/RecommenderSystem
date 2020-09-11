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
    # spark = (SparkSession.builder
    #          .master("local")
    #          .appName("TandonProud_als")
    #          .config("spark.executor.memory", "5g")
    #          .config("spark.driver.memory", "5g")
    #          .getOrCreate())
    # spark.sparkContext.setLogLevel("ERROR")
    if len(sys.argv) > 0:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        split(input_file, output_file)




# def split(filehandler, outputfile):
#     df = pd.read_csv(filehandler)
#     train = df.sample(frac=0.6, random_state=1)   # Here, we cannot just sample it with a percent.
#     test_vali = df.loc[~df.index.isin(train.index)] # We need to follow the ratio of different user_id
#     test = test_vali.sample(frac=0.5, random_state=1)
#     validation = test_vali.loc[~test_vali.index.isin(test.index)]
#
#     train.to_csv(f'{outputfile}/train.csv', index=False)
#     test.to_csv(f'{outputfile}/test_temp.csv', index=False)
#     validation.to_csv(f'{outputfile}/validation_temp.csv', index=False)
#
#     write_csv(f'{outputfile}/train.csv', f'{outputfile}/test_temp.csv', f'{outputfile}/test.csv')
#     write_csv(f'{outputfile}/train.csv', f'{outputfile}/validation_temp.csv', f'{outputfile}/validation.csv')


# def write_csv(train_csv, temp_csv, output_csv):
#     train_write = csv.writer(open(train_csv, 'a'))
#     reader = csv.reader(open(temp_csv, 'r'), delimiter=",")
#     dic = count_interaction_2(temp_csv)   # test_temp字典保存所有id的数值
#     empty_dic = {}                           # 空字典 记录当前写了多少行 是否达到0.6
#     header = next(reader)
#     test_writer = csv.writer(open(output_csv, 'w'))    # 刚开始可以进行6 4分 重点是对之后的0.4部分进行依据id的对半分  test
#     test_writer.writerow(header)
#     for i, row in enumerate(reader):
#         if empty_dic.get(row[0],0) <= dic[row[0]]*0.5:
#             train_write.writerow(row)
#         else:
#             test_writer.writerow(row)
#         empty_dic[row[0]] = empty_dic.get(row[0],0) + 1
#
#
# def subsample(train, dataset, output_file, filename, config):
#     if config:
#         train.to_csv(f'{output_file}/train.csv', index=False)
#     dataset.to_csv(f'{output_file}/{filename}_temp.csv', index=False)
#
#     dict_store = count_interaction_2(open(f'{output_file}/{filename}_temp.csv', 'r'))
#     split_1 = csv.writer(open(f'{output_file}/train.csv', 'a'))
#     split_2 = csv.writer(open(f'{output_file}/{filename}.csv', 'w'))
#     temp = csv.reader(open(f'{output_file}/{filename}_temp.csv', 'r'))
#     header = next(temp)
#     split_2.writerow(header)
#     cur_dic = {}
#     for i, row in enumerate(temp):
#         if cur_dic.get(row[0], 0) < dict_store[row[0]]//2:
#             split_1.writerow(row)
#         else:
#             split_2.writerow(row)
#         cur_dic[row[0]] = cur_dic.get(row[0], 0) + 1


#
# def count_interaction_2(filehandler, delimiter=','):
#     reader = csv.reader(open(filehandler, 'r'), delimiter=delimiter)
#     store = {}
#     for i, row in enumerate(reader):
#         store[row[0]] = store.get(row[0],0) + 1
#     return store

# def concat(train, filehandler):  hdfs:/users/zz2671/final_project/small_interactions.csv
#     temp = pd.read_csv(filehandler)
#     opt = pd.concat([train, temp])
#     return opt

# hdfs://dumbo/user/zz2671/final_project/small_interactions.csv

# def count(filehandler):
#     reader = csv.reader(filehandler, delimiter=',')
#     num = 0
#     for i, row in enumerate(reader):
#         num += 1
#     return num
#
# def split(filehandler, row_limit, delimiter=',' ,
#     output_name_template='output_%s.csv', output_path='.', keep_headers=True):
#     """
#     Splits a CSV file into multiple pieces.
#
#     A quick bastardization of the Python CSV library.
#     Arguments:
#         `row_limit`: The number of rows you want in each output file
#         `output_name_template`: A %s-style template for the numbered output files.
#         `output_path`: Where to stick the output files
#         `keep_headers`: Whether or not to print the headers in each output file.
#     Example usage:
#
#         >> from toolbox import csv_splitter;
#         >> csv_splitter.split(csv.splitter(open('/home/ben/Desktop/lasd/2009-01-02 [00.00.00].csv', 'r')));
#
#     """
#     reader = csv.reader(filehandler, delimiter=delimiter)
#     current_piece = 1
#     current_out_path = os.path.join(
#          output_name_template  % current_piece
#     )
#     current_out_writer = csv.writer(open(current_out_path, 'w'))
#     current_limit = int(row_limit * 0.6)
#     if keep_headers:
#         headers = next(reader)
#         current_out_writer.writerow(headers)
#     for i, row in enumerate(reader):
#         if i + 1 > current_limit:
#             current_piece += 1
#             current_limit += int(row_limit * 0.2)
#             current_out_path = os.path.join(
#                output_name_template  % current_piece
#             )
#             current_out_writer = csv.writer(open(current_out_path, 'w'))
#             if keep_headers:
#                 current_out_writer.writerow(headers)
#         current_out_writer.writerow(row)
#
#
# row_limit = count(open('interaction.csv'))
# split(open('interaction.csv'), row_limit)

    # reader_test = csv.reader(open(f'{outputfile}/test_temp.csv', 'r'), delimiter=delimiter)
    # dic = count_interaction_2(f'{outputfile}/test_temp.csv')  # test_temp字典保存所有id的数值
    # empty_dic = {}  # 空字典 记录当前写了多少行 是否达到0.6
    # header = next(reader_test)
    # test_writer = csv.writer(open(f'{outputfile}/test.csv', 'w'))  # 刚开始可以进行6 4分 重点是对之后的0.4部分进行依据id的对半分  test
    # test_writer.writerow(header)
    # for i, row in enumerate(reader_test):
    #     if empty_dic.get(row[0], 0) <= dic[row[0]] * 0.5:
    #         train_write.writerow(row)
    #     else:
    #         test_writer.writerow(row)
    #
    # dic.clear()
    # empty_dic.clear()
