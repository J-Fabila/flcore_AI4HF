import json
import numpy as np
import pandas as pd
import argparse
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

from utils import Parameters

def KS_test(old_data,new_data,feature,alpha=0.05):
    # Kolmogorov-Smirnov (KS test)
    # For Continous variables
    empty_1 = old_data[feature].notna().any()
    empty_2 = new_data[feature].notna().any()
    if empty_1 == True and empty_2 == True:
        old_data["temp"] = pd.to_numeric(old_data[feature], errors="coerce").fillna(0)
        #print("NANS DETECTADO", dat_1["temp"].isna().sum())
        new_data["temp"] = pd.to_numeric(new_data[feature], errors="coerce").fillna(0)

        ks_stat, p_value = ks_2samp(old_data["temp"], new_data["temp"])
        if p_value < alpha:
            return 2 # Data drift detected
        elif p_value < 1.0 and p_value > alpha:
            return 1 # Changes but not a drift
        else:
            return 0 # No changes
    elif empty_1 == False and empty_2 == False:
        return 3 # both empty, thus, no changes, thus, no data drift
    else:
        return 4  # one is empty and the other no, thus, there were changes, thus, there are important changes in distribution

def chi2(old_data, new_data, feature, alpha=0.05):
    empty_1 = old_data[feature].notna().any()
    empty_2 = new_data[feature].notna().any()
    if empty_1 == True and empty_2 == True:
        old = old_data[[feature]].copy()
        old["source"] = "old"
        new = new_data[[feature]].copy()
        new["source"] = "new"
        combined = pd.concat([old, new], axis=0)
        combined = combined.dropna(subset=[feature])
        contingency = pd.crosstab(combined[feature], combined["source"])
        stat, p_value, _, _ = chi2_contingency(contingency)
        if p_value < alpha:
            return 2 # Data drift
        elif p_value < 1.0 and p_value > alpha:
            return 1 # Changes but no data drift
        else:
            return 0 # No changes
    elif empty_1 == False and empty_2 == False:
        return 3 # both empty, thus, no changes, thus, no data drift
    else:
        return 4  # one is empty and the other no, thus, there were changes, thus, there are important changes in distribution

def drift_detection(config):

    data_file = config["data_file_1"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat_1 = pd.read_parquet(data_file)
    elif ext == "csv":
        dat_1 = pd.read_csv(data_file)


    data_file = config["data_file_2"]
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
        # IMPORTANT: DATE TIMES ARE NOT ANALYZED
        if feat["dataType"] == "NUMERIC":
            drift = KS_test(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "NOMINAL":
            drift = chi2(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "BOOLEAN":
            drift = chi2(dat_1,dat_2,feature,config["alpha"])
        #print("FEATURE",feature, feat["dataType"], drift)
        drift_dict[feature] = drift

    for feat in metadata["entity"]["outcomes"]:
        feature = feat["name"]
        if feat["dataType"] == "NUMERIC":
            drift = KS_test(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "NOMINAL":
            drift = chi2(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "BOOLEAN":
            drift = chi2(dat_1,dat_2,feature,config["alpha"])
        #print("OUTCOME",feature, feat["dataType"], drift)
        drift_dict[feature] = drift

    # Se tendría que añadir también el p_value por cada variable, pero será para
    # la siguiente iteración

    with open("data_drift.json", "w") as json_file_out:
        json.dump(drift_dict, json_file_out, indent=4)
    # se podría hacer que devuelva varios valores:
    # 0 : sin cambios
    # 1 : cambio en los valores pero sin data drift
    # 2 : data drift
    # 3 : both empty, thus, no changes, thus, no data drift
    # 4 : one is empty and the other no, thus, there were changes, thus, there are important changes in distribution

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    parser.add_argument("--data_file_1", type=str, default="" , help="Data file 1")
    parser.add_argument("--data_file_2",type=str, default="", help="Data file 2")
    parser.add_argument("--metadata_file_1", type=str, default="", help="metadata file 1")
    parser.add_argument("--alpha", type=float, default=0.05 , help="alpha threshold")

    args = parser.parse_args()

    config = vars(args)

    drift_detection(config)
    # Aquí se tendría que integrar con la plataforma