# Utiliza una imagen base de Python para Windows
FROM python:3.9-windowsservercore

# Instala PyTorch y sus dependencias
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instala las demás dependencias
RUN pip install wandb rich tqdm flwr pytorch-lightning

# Establece el directorio de trabajo en /workspace
WORKDIR /workspace

# Copia todos los archivos del contexto de construcción al directorio de trabajo en el contenedor
COPY . .

# Cambia al archivo de configuración necesario
CMD ["python", "script.py", "configs/client_1.yaml"]

#docker build -t mi-imagen .

#docker run -v ruta/a/tu/Federado:/Federado mi-imagen