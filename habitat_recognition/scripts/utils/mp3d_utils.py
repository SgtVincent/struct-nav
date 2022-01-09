# matterport dataset utils
import os 
import csv 
import pandas as pd 

def read_label_mapping(filename, label_from='mpcat40', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    return mapping

