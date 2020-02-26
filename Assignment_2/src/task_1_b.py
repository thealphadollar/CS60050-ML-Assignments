from csv import DictReader, DictWriter
from os import path
from collections import defaultdict
from math import floor

# set directory for data
src_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'data')

# dictionary to load data into
dictionary_red_wine = []

# default dict yields a default value if the key is not present
sum_dict = defaultdict(lambda: 0)
sq_sum_dict = defaultdict(lambda: 0)
min_dict = defaultdict(lambda: +1e10)
max_dict = defaultdict(lambda: -1e10)
total_vals = 0

with open(path.join(src_dir, 'winequality-red.csv'), 'r') as f_in:
    # the delimiter for the data is semi-colon in the CSV
    data_red_wine = DictReader(f_in, delimiter=';')

    # iterating every entry in the data
    for index, vals in enumerate(data_red_wine):
        dictionary_red_wine.append(vals)
        for key in vals.keys():
            # handle quality as per the given conditions
            if key == 'quality':
                if int(vals['quality'])<4:
                    dictionary_red_wine[index]['quality'] = 0
                elif int(vals['quality'])>6:
                    dictionary_red_wine[index]['quality'] = 2
                else:
                    dictionary_red_wine[index]['quality'] = 1
            # calculate sum and sum of square of values for that key
            else:
                sum_dict[key] = sum_dict[key] + float(vals[key])
                sq_sum_dict[key] = sq_sum_dict[key] + (float(vals[key])*float(vals[key]))
        total_vals += 1

mean_dict = dict()
sd_dict = dict()

for key in sum_dict.keys():
    # calculate mean for each key
    mean_dict[key] = sum_dict[key] / total_vals
    # calculate standard deviation for each key
    sd_dict[key] = pow((sq_sum_dict[key]+(total_vals*pow(mean_dict[key],2))-(2*mean_dict[key]*sum_dict[key])) / total_vals, 0.5)

# apply scaling
for index, vals in enumerate(dictionary_red_wine):
    for key in vals.keys():
        # exclude quality attribute
        if key != 'quality':
            # apply z score scaling := (value - mean) / standard_deviation
            dictionary_red_wine[index][key] = (float(dictionary_red_wine[index][key]) - mean_dict[key])/sd_dict[key]
            # calculating the max and min z-score for each column
            min_dict[key] = min(min_dict[key], dictionary_red_wine[index][key])
            max_dict[key] = max(max_dict[key], dictionary_red_wine[index][key])

# division into bucket
for index, vals in enumerate(dictionary_red_wine):
    for key in vals.keys():
        # exclude quality key
        if key != 'quality':
            # take the difference between the max and min
            diff = max_dict[key] - min_dict[key]
            # divide the difference by 4
            divDiff = diff / 4
            # the value lies in the floor (except last bucket) of difference from min divided by divDiff
            dictionary_red_wine[index][key] = floor((dictionary_red_wine[index][key] - min_dict[key]) / divDiff)
            if dictionary_red_wine[index][key] == 4:
                dictionary_red_wine[index][key] = 3

with open(path.join(src_dir, 'dataset_B.csv'), 'w') as f_out:
    writer = DictWriter(f_out, fieldnames=dictionary_red_wine[0].keys(), delimiter=';')
    writer.writeheader()
    writer.writerows(dictionary_red_wine)