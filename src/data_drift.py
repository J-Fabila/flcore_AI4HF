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
        print("Drift detectado")
        return True
    else:
        return False

def chi2(old_data,new_data,feature):
    # Chi-squared
    # For categorical variables
    contingency_table = pd.crosstab(old_data[feature], new_data[feature])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    if p < 0.05:
        print("Drift detectado")
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

    # HASTA AQUI BIEN, lo demás habrá que cambiarlo
    # creo que si necesitariamos tener el tipo de dato del metadata, solo de uno

    for variables:
        if tipo == nominal:
            pass

def load_dt4h(config,id):
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

    """    # Print statistics
    for i in dat.keys():
        maxim = dat[i].max()
        minim = dat[i].min()
        mean = dat[i].mean()
        estd = dat[i].std()
        print(f"Column: {i}")
        print(f"  Maximum:          {maxim:10.2f}")
        print(f"  Minimum:          {minim:10.2f}")
        print(f"  Mean:             {mean:10.2f}")
        print(f"  Std dev:          {estd:10.2f}")
        print("-" * 40)
    """

    dat_shuffled = dat.sample(frac=1).reset_index(drop=True)

    target_labels = config["target_label"]
    train_labels = config["train_labels"]
    data_train = dat_shuffled[train_labels] #.to_numpy()
    data_target = dat_shuffled[target_labels] #.to_numpy()

    X_train = data_train[:int(dat_len*config["train_size"])]
    y_train = data_target[:int(dat_len*config["train_size"]):].iloc[:, 0]

    X_test = data_train[int(dat_len*config["train_size"]):]
    y_test = data_target[int(dat_len*config["train_size"]):].iloc[:, 0]
    return (X_train, y_train), (X_test, y_test)

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