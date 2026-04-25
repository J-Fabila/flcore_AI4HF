import torch
import pytorch_lightning as pl
from dataloaders import MMsDataSet, LightningWrapperData  # Suponiendo que tienes definidos tus dataloaders

def test(modelo, dataset):
    """
    Cosas por hacer:
    * Implementar el accuracy en el model: esta en STEP del modelwrapper
    """
    print(" ************************************************ NEW TESTING ")
    print(" DATA SET CARGADO")
    # Crea un DataLoader para el conjunto de datos de prueba
    print(" test loader")
    # Inicializa el evaluador de PyTorch Lightning
    #trainer = pl.Trainer()
    
    trainer = pl.Trainer(
    max_epochs=modelo.config['epochs'],
    num_nodes=modelo.config['n_gpu_nodes'],
    default_root_dir=modelo.config['log_path'],
    #callbacks=[model_checkpoint ,lr_monitor,rich_progress_bar],
    logger=False, #_logger,
    gradient_clip_val=modelo.config['clip'],
    log_every_n_steps = 1
    )

    print(" TRAINER DEFINIDO")
    # Evalúa el modelo en el conjunto de datos de prueba
    result = trainer.test(modelo,dataset)
    print(" =========================================")
    print(" RESULT ", type(result), result)
#   RESULT  <class 'list'> [{'test_loss': 1.5813443660736084}]
    # Imprime los resultados
    #print(result)
    loss = result[0]["test_loss"]
    accuracy = result[0]["acc"]
    print("****************************************************TERMINA TESTING NUEVO")
    return loss, accuracy