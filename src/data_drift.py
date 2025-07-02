import json
import numpy as np
import pandas as pd

import scikitlearn

def drift_detection(config):
    with open(config["data_path"]+config['metadata_file'], 'r') as file:
        metadata = json.load(file)

    data_file = config["data_path"] + config["data_file"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat = pd.read_parquet(data_file)
    elif ext == "csv":
        dat = pd.read_csv(data_file)

    dat_len = len(dat)

    # Numerical variables
    numeric_columns_non_zero = {}
    for feat in metadata["entries"][0]["featureSet"]["features"]:
        if feat["dataType"] == "NUMERIC" and feat["statistics"]["numOfNotNull"] != 0:
            # statistic keys = ['Q1', 'avg', 'min', 'Q2', 'max', 'Q3', 'numOfNotNull']
            numeric_columns_non_zero[feat["name"]] = (
                feat["statistics"]["Q1"],
                feat["statistics"]["avg"],
                feat["statistics"]["min"],
                feat["statistics"]["Q2"],
                feat["statistics"]["max"],
                feat["statistics"]["Q3"],
                feat["statistics"]["numOfNotNull"],
            )

    for col, (q1,avg,mini,q2,maxi,q3,numOfNotNull) in numeric_columns_non_zero.items():
        if col in dat.columns:
            if config["normalization_method"] == "IQR":
               dat[col] = iqr_normalize(dat[col], q1,q2,q3 )
            elif config["normalization_method"] == "STD":
                pass # no std found in data set
            elif config["normalization_method"] == "MIN_MAX":
               dat[col] = min_max_normalize(col, mini, maxi)
    tipos=[]
    map_variables = {}
    for feat in metadata["entries"][0]["featureSet"]["features"]:
        tipos.append(feat["dataType"])
        if feat["dataType"] == "NOMINAL" and feat["statistics"]["numOfNotNull"] != 0:
            num_cat = len(feat["statistics"]["valueset"])
            map_cat = {}
            for ind, cat in enumerate(feat["statistics"]["valueset"]):
                map_cat[cat] = ind
            map_variables[feat["name"]] = map_cat
    for col,mapa in map_variables.items():
        dat[col] = dat[col].map(mapa)
    
    dat[map_variables.keys()].dropna()
    
    tipos=[]
    map_variables = {}
    boolean_map = {np.bool_(False) :0, np.bool_(True):1, "False":0,"True":1}
    for feat in metadata["entries"][0]["featureSet"]["features"]:
        tipos.append(feat["dataType"])
        if feat["dataType"] == "BOOLEAN" and feat["statistics"]["numOfNotNull"] != 0:
            map_variables[feat["name"]] = boolean_map
    for col,mapa in map_variables.items():
        dat[col] = dat[col].map(boolean_map)
    
    dat[map_variables.keys()].dropna()

    return 