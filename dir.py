import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "genero": ["hombre", "mujer", "hombre", "mujer", "hombre", "mujer", "hombre", "mujer"],
    "ingreso": [50000, 30000, 52000, 28000, 48000, 31000, 51000, 29000]
})

def apply_disparate_impact_remover(df, features, sensitive_attr):
    df = df.copy()
    print("apply DIR", features)
    for feature in features:
        print("feature", feature)
        for value in df[sensitive_attr].unique():
            print("value", value)
            mask = df[sensitive_attr] == value
            group_mean = df.loc[mask, feature].mean()
            group_std = df.loc[mask, feature].std()
            if group_std > 0:
                df.loc[mask, feature] = (df.loc[mask, feature] - group_mean) / group_std
    return df

df_normalizado = apply_disparate_impact_remover(df, ["ingreso"], "genero")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(x="genero", y="ingreso", data=df, ax=axes[0])
axes[0].set_title("Ingreso original por género")

sns.boxplot(x="genero", y="ingreso", data=df_normalizado, ax=axes[1])
axes[1].set_title("Ingreso normalizado por género (DIR)")

plt.tight_layout()
plt.show()
