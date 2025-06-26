import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample

def apply_reweighing(df, sensitive_attr, label_attr):
    df = df.copy()
    joint = df.groupby([sensitive_attr, label_attr]).size() / len(df)
    sensitive_marginal = df.groupby(sensitive_attr).size() / len(df)
    label_marginal = df.groupby(label_attr).size() / len(df)

    def compute_weight(row):
        s = row[sensitive_attr]
        y = row[label_attr]
        expected = sensitive_marginal[s] * label_marginal[y]
        actual = joint[s][y]
        return expected / actual if actual > 0 else 1.0

    df["sample_weight"] = df.apply(compute_weight, axis=1)
    return df

def apply_disparate_impact_remover(df, features, sensitive_attr):
    df = df.copy()
    for feature in features:
        for value in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == value
            group_mean = df.loc[mask, feature].mean()
            group_std = df.loc[mask, feature].std()
            if group_std > 0:
                df.loc[mask, feature] = (df.loc[mask, feature] - group_mean) / group_std
    return df

def apply_balanced_sampling(df, sensitive_attr):
    # Identifica el grupo minoritario y mayoritario
    groups = df[sensitive_attr].value_counts()
    max_n = groups.max()
    
    dfs = []
    for group_val in groups.index:
        df_group = df[df[sensitive_attr] == group_val]
        df_resampled = resample(df_group, replace=True, n_samples=max_n, random_state=42)
        dfs.append(df_resampled)
    
    df_balanced = pd.concat(dfs).reset_index(drop=True)
    return df_balanced

def apply_fair_pca(df, features, sensitive_attr, n_components=5):
    df = df.copy()
    X = df[features].copy()
    
    # Centrado por grupo sensible
    for group_val in df[sensitive_attr].unique():
        mask = df[sensitive_attr] == group_val
        group_mean = X.loc[mask].mean()
        X.loc[mask] = X.loc[mask] - group_mean

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Reemplaza las features originales con componentes principales
    for i in range(X_pca.shape[1]):
        df[f"fair_pca_{i+1}"] = X_pca[:, i]
    df.drop(columns=features, inplace=True)
    
    return df

def unbias_preprocessing(config):
    sensitive_attr = config["sensitive"]
    label_attr = config["label"]

    data_file = config["data_path"] + config["data_file"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat = pd.read_parquet(data_file)
    elif ext == "csv":
        dat = pd.read_csv(data_file)

    features = [col for col in dat.columns if col not in [sensitive_attr, label_attr]]

    method = config["method"]

    if method == "SUP":
        dat = dat.drop(columns=sensitive_attr)
    elif method == "RW":
        dat = apply_reweighing(dat, sensitive_attr, label_attr)
    elif method == "DIR":
        dat = apply_disparate_impact_remover(dat, features, sensitive_attr)
    elif method == "SAM":
        dat = apply_balanced_sampling(dat, sensitive_attr)
    elif method == "FPCA":
        dat = apply_fair_pca(dat, features, sensitive_attr, n_components=config.get("n_components", 5))

    preprocessed_file = "archivo.parquet"
    dat.to_parquet(preprocessed_file, engine="pyarrow")
    return preprocessed_file
