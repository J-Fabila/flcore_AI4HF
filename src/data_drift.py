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

    with open(config["data_path_1"]+config['metadata_file_1'], 'r') as file:
        metadata = json.load(file)

    for feat in metadata["entity"]["features"]:
        feature = feat["name"]
        if feat["dataType"] == "NUMERIC":
            KS_test(dat_1,dat_2,feature)
        elif feat["dataType"] == "NOMINAL":
            chi2(dat_1,dat_2,feature)
        elif feat["dataType"] == "BOOLEAN":
            chi2(dat_1,dat_2,feature)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--metadata_file", type=str, default="metadata.json", help="Json file with metadata")
    parser.add_argument("--data_file", type=str, default="data.parquet" , help="parquet o csv file with actual data")
    parser.add_argument("--normalization_method",type=str, default="IQR", help="Type of normalization: IQR STD MIN_MAX")
    parser.add_argument("--train_labels", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--target_label", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--train_size", type=float, default=0.8, help="Fraction of dataset to use for training. [0,1)")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode")
    parser.add_argument("--node_name", type=str, default="./", help="Node name for certificates")
    parser.add_argument("--sandbox_path", type=str, default="./", help="Sandbox path to use")

    args = parser.parse_args()

    config = vars(args)

    drift_detection(config)
    # Aquí se tendría que integrar con la plataforma