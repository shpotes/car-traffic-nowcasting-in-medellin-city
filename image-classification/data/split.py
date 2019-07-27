import os
import pandas as pd


def split(x, test_split=15, val_split=15):
    x = hash(x) % 100
    
    if x < test_split:
        return 'test'
    elif x < test_split + val_split:
        return 'val'
    else:
        return 'train'


filename = pd.Series(os.listdir())
filename = filename[filename.apply(lambda x: '.jpg' in x)]

split = filename.apply(split)

metadata = pd.DataFrame([filename, split]).T
metadata.columns = ['filename', 'split']

for i in metadata.iterrows():
    tmp = i[1]
    os.rename(tmp.filename, tmp.split + '/' + tmp.filename)

metadata.to_csv('metadata.csv')
