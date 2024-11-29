# Federated learning platform

## Environment Requirements

To set up and run the project, follow the steps below:

1. **Install Conda (optional)**  
   While not mandatory, it is recommended to use a Conda environment to manage dependencies and avoid conflicts.

2. **Install Dependencies with pip**  
   Run the following command to install the required libraries:
   ```bash
   pip install wandb rich tqdm flwr pytorch-lightning scikit-learn torchdiffeq lifelines pyarrow
   ```

3. **Install PyTorch**  
   PyTorch must be installed according to your computer's specifications (operating system, Python version, and hardware availability, such as GPU). Refer to the official PyTorch installation guide at [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage Instructions

To run the project, use the `main.py` script, specifying the configuration file and the desired role (server or client). Follow these steps:

1. **Run the Server**
   ```
   python main.py configs/config.file server
   ```

2. **Run the Client (MLP)**
   ```
   python main.py configs/config.file MLP
   ```

Ensure the configuration file (`configs/config.file`) is properly set up for your environment and use case.
