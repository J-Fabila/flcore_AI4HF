import pandas as pd
import numpy as np

def apply_reweighing(df, sensitive_attr, label_attr):
    df = df.copy()
    # Calcular la distribución conjunta y marginal
    joint = df.groupby([sensitive_attr, label_attr]).size() / len(df)
    sensitive_marginal = df.groupby(sensitive_attr).size() / len(df)
    label_marginal = df.groupby(label_attr).size() / len(df)

    # Calcular los pesos
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
    # Aqui se tendria que verificar que las variables sean de tipo numéricas
    # Normalizar características por grupo sensible (media 0, varianza 1)
    for feature in features:
        for value in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == value
            group_mean = df.loc[mask, feature].mean()
            group_std = df.loc[mask, feature].std()
            if group_std > 0:
                df.loc[mask, feature] = (df.loc[mask, feature] - group_mean) / group_std
    return df

def unbias_preprocessing(config):
    # A cambiar por los nombres de las variables reales
    # Añadir las variables al config y al utils
    sensitive_attr = config["sensitive"]
    label_attr = config["label"]

    data_file = config["data_path"] + config["data_file"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat = pd.read_parquet(data_file)
    elif ext == "csv":
        dat = pd.read_csv(data_file)
    # dat está listo para usarse
    dat_len = len(dat)

    features = [col for col in dat.columns]

#    assert sensitive_attr in dat.columns and label_attr in dat.columns

    if config["method"] == "SUP":
        # Faltaría verificar si se puede con varios atributos
        dat = dat.drop(columns=sensitive_attr)
    elif config["method"] == "RW":
        dat = apply_reweighing(dat, sensitive_attr, label_attr)
    elif config["method"] == "DIR":
        dat = apply_disparate_impact_remover(dat, [label_attr],sensitive_attr)

    preprocessed_file = "archivo.parquet"
    dat.to_parquet(preprocessed_file, engine="pyarrow")
    return preprocessed_file