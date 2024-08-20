FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN pip install wandb rich tqdm flwr pytorch-lightning 

WORKDIR /workspace

COPY * /workspace/

# Cambiar por el config file necesario
CMD ["python", "script.py", "configs/client_1.yaml"]