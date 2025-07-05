import json
import numpy as np
import pandas as pd
import argparse

from utils import Parameters

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


"""
# NOTAS
a. Kolmogorov-Smirnov (KS test)

Para variables continuas.

Compara dos distribuciones (entrenamiento vs. producción).

from scipy.stats import ks_2samp

ks_stat, p_value = ks_2samp(train_data['edad'], prod_data['edad'])
if p_value < 0.05:
    print("Drift detectado")
b. Chi-cuadrado
Para variables categóricas.

from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(train_data['genero'], prod_data['genero'])
chi2, p, _, _ = chi2_contingency(contingency_table)
if p < 0.05:
    print("Drift detectado")
"""