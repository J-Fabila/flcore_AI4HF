from pycox.models import DeepSurv
from pycox.datasets import metabric
from torchtuples import MLPVanilla

# Cargar dataset
df = metabric.read_df()
x = df.drop(columns=['duration', 'event'])
y = (df['duration'].values, df['event'].values)

# Definir red neuronal
net = MLPVanilla(in_features=x.shape[1], num_nodes=[32, 32], out_features=1)
model = DeepSurv(net)

# Preparar datos
model.fit(x, y, batch_size=128, epochs=10)

# Predicción de riesgo
pred_risk = model.predict(x)
