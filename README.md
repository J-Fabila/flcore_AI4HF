# Federated learning platform

## Environment Requirements

To set up and run the project, follow the steps below:

1. **Install Conda (optional)**  
   While not mandatory, it is recommended to use a Conda environment to manage dependencies and avoid conflicts.
   ```
   conda create -n mlp python=3.9
   ```

2. **Install Dependencies with pip**  
   Run the following commands to activate and install the required libraries:
   ```bash
   conda activate mlp
   pip install wandb rich tqdm flwr==1.8 pytorch-lightning scikit-learn torchdiffeq lifelines pyarrow
   ```

3. **Install PyTorch**  
   PyTorch must be installed according to your computer's specifications (operating system, Python version, and hardware availability, such as GPU). Refer to the official PyTorch installation guide at [pytorch.org](https://pytorch.org/get-started/locally/). For example, with CUDA 12.4 in Linux:
   ```
   pip3 install torch torchvision torchaudio
   ```
## Usage Instructions

To run the project, use the `main.py` script, specifying the configuration file and the desired role (server or client). Follow these steps:

1. **Run the Server**
   ```
   python main.py configs/server.yaml
   ```

2. **Run the Client (MLP)**
   ```
   python main.py configs/MLP.yaml
   python main.py configs/MLP_2.yaml
   ```

Ensure the configuration file (`configs/config.file`) is properly set up for your environment and use case.
