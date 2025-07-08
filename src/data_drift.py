import json
import numpy as np
import pandas as pd
import argparse
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

from utils import Parameters

def KS_test(old_data,new_data,feature):
    # Kolmogorov-Smirnov (KS test)
    # For Continous variables
    ks_stat, p_value = ks_2samp(old_data[feature], new_data[feature])
    if p_value < 0.05:
        print("Drift detected in",feature)
        return True
    else:
        return False

def chi2(old_data,new_data,feature):
    # Chi-squared
    # For categorical variables
    contingency_table = pd.crosstab(old_data[feature], new_data[feature])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    if p < 0.05:
        print("Drift detected in ",feature)
        return True
    else:
        return False

def drift_detection(config):

    data_file = config["data_path_1"] + config["data_file_1"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat_1 = pd.read_parquet(data_file)
    elif ext == "csv":
        dat_1 = pd.read_csv(data_file)


    data_file = config["data_path_2"] + config["data_file_2"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat_2 = pd.read_parquet(data_file)
    elif ext == "csv":
        dat_2 = pd.read_csv(data_file)

    with open(config['metadata_file_1'], 'r') as file:
        metadata = json.load(file)

    drift_dict = {}
    for feat in metadata["entity"]["features"]:
        feature = feat["name"]
        print("FEATURE",feature, feat["dataType"])
        if feat["dataType"] == "NUMERIC":
            dat_1["temp"] = pd.to_numeric(dat_1[feature], errors="coerce")
            #print("NANS DETECTADO", dat_1["temp"].isna().sum())

            dat_1["temp"].fillna(0)

            dat_2["temp"] = pd.to_numeric(dat_2[feature], errors="coerce")
            dat_2["temp"].fillna(0)

            drift = KS_test(dat_1,dat_2,"temp")
        elif feat["dataType"] == "NOMINAL":
            empty_1 = dat_1[feature].notna().any()
            empty_2 = dat_2[feature].notna().any()
            print("NOT NA",empty_1, empty_2) # if false significa que esta vacia
            if empty_1 == True and empty_2 == True: # ambos false
                drift = chi2(dat_1,dat_2,feature)
            elif empty_1 == False and empty_2 == False: # ambos true
                drift = False # both empty, thus, no changes, thus, no data drift
            else: # one is empty and the other no, thus, there were changes, thus, there is data drift
                drift = True
        elif feat["dataType"] == "BOOLEAN":
            empty_1 = dat_1[feature].notna().any()
            empty_2 = dat_2[feature].notna().any()
            print("NOT NA",empty_1, empty_2) # if false significa que esta vacia
            if empty_1 == True and empty_2 == True: # ambos false
                drift = chi2(dat_1,dat_2,feature)
            elif empty_1 == False and empty_2 == False: # ambos true
                drift = False # both empty, thus, no changes, thus, no data drift
            else: # one is empty and the other no, thus, there were changes, thus, there is data drift
                drift = True
    # ¿change to json?
    with open("archivo.json", "w") as json_file_out:
        json.dump(drift_dict, json_file_out, indent=4)
        
    #return drift_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    parser.add_argument("--data_path_1", type=str, default="", help="Data path 1")
    parser.add_argument("--data_path_2", type=str, default="", help="Data path 2")
    parser.add_argument("--data_file_1", type=str, default="" , help="Data file 1")
    parser.add_argument("--data_file_2",type=str, default="", help="Data file 2")
    parser.add_argument("--metadata_file_1", type=str, default="", help="metadata file 1")

    args = parser.parse_args()

    config = vars(args)

    drift_detection(config)
    # Aquí se tendría que integrar con la plataforma