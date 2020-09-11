import csv

def count(filehandler, delimiter=','):
    reader = csv.reader(filehandler, delimiter=delimiter)
    store = set()
    cur_id = ""
    num = 0
    for i, row in enumerate(reader):
        if row[0] != cur_id:
            if num > 9:
                store.add(cur_id)
            cur_id = row[0]
            num = 1
        else:
            num += 1
    return store


def filter(filehandler, store, delimiter=','):
    reader = csv.reader(filehandler, delimiter=delimiter)
    header = next(reader)
    current_out_writer = csv.writer(open('interaction_split.csv', 'w'))
    current_out_writer.writerow(header)
    for i, row in enumerate(reader):
        if row[0] in store:
            current_out_writer.writerow(row)


store = count(open('interaction.csv'))
filter(open('interaction.csv'), store, delimiter=',')