import csv

def count(filehandler, delimiter=','):
    reader = csv.reader(filehandler, delimiter=delimiter)
    store = {}
    for i, row in enumerate(reader):
        store[row[0]] = store.get(row[0],0) + 1
    return store


def spliiter(filehandler, store, delimiter=','):
    reader = csv.reader(filehandler, delimiter=delimiter)
    split_1 = csv.writer(open('split_1.csv', 'w'))
    split_2 = csv.writer(open('split_2.csv', 'w'))
    header = next(reader)
    split_1.writerow(header)
    split_2.writerow(header)
    cur_id = ''
    num = 0
    for i, row in enumerate(reader):
        if row[0] == cur_id:
            if num < store[row[0]]//2:
                split_1.writerow(row)
            else:
                split_2.writerow(row)
            num += 1
        else:
            cur_id = row[0]
            num = 1
            split_1.writerow(row)


store = count(open('output_2.csv','r'))
spliiter(open('output_2.csv', 'r'), store)