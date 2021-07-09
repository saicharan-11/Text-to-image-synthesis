# -*- coding: utf-8 -*-
from zipfile import ZipFile
from pandas import read_csv
from io import BytesIO
import numpy as np
from pickle import dump

archive = ZipFile(file="flickr_dataset.zip", mode="r")

df = read_csv("captions.csv", sep=",")
df = df.rename(columns=lambda x: x.strip())

IMAGELIST = df.loc[:, "image"].unique()

captions = {}

for (idx, label) in enumerate(IMAGELIST):
    captions[idx] = list(df.loc[df["image"] == label, "caption"].values)
    
with open('captions.pickle', 'wb') as file:
    dump(captions, file)

del file

result = [captions[key] for key in sorted(captions.keys())]
with open('captions.pickle', 'wb') as file:
    dump(result, file)

del file