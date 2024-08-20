FROM python:3.9-slim

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install wandb rich tqdm flwr pytorch-lightning 


WORKDIR /workspace
COPY * /workspace/

# Cambiar por el config file necesario
CMD ["python", "script.py", "configs/client_1.yaml"]