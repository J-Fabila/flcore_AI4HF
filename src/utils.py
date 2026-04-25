### ¿Utilizarmos hydra o será mejor a la vieja usanza?
import yaml

import os
import re
import glob



def get_parameters(directory):
    ckpt_files = glob.glob(os.path.join(directory, '*.ckpt'))

    best_checkpoint = None
    best_val_loss = float('inf')

    for ckpt_file in ckpt_files:
        match = re.search(r'epoch=(\d+)-val_loss=([\d.]+)\.ckpt', ckpt_file)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt_file
    # Prepares filename for the next iteration
    new_filename = "model_client_0_round_0.ckpt"
    os.rename(best_checkpoint, os.path.join(directory,  new_filename))
