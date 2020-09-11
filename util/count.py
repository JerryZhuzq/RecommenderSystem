import csv

reader = csv.reader(open('interaction_split.csv'), delimiter=',')
num = 0
for i, row in enumerate(reader):
    num += 1
print(num)