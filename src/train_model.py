import hydra
import logging

from numpy.random import seed
from hydra import utils
from omegaconf.dictconfig import DictConfig

from utils import (
    model_data_preprocessing,
    save_model,
    metrics,
    model_train,
    preprocessing_features,
    read_data
)


@hydra.main(config_path="../config", config_name="config_file.yaml", version_base="1.1")
def run_training(config: DictConfig):
    """
    Método general. Entrenamiento de un modelo de Machine Learning
    Uso de ficheros de configuración para realizar el proceso de aprendizaje
    """
    logging.info("EMPEZAMOS...")
    seed(config.default.seed)  # default seed for numpy

    # path del código fuente
    current_path = utils.get_original_cwd()

    # directorio, nombre y formato donde se encuentran los datos
    data_path = config.dataset.path
    data_name = config.dataset.data
    data_format = config.dataset.format

    data_path = f"{current_path}/{data_path}/{data_name}.{data_format}"
    datos = read_data(config, data_format, data_path)
    if datos.empty:
        raise TypeError("Format file is not `csv`")

    datos = preprocessing_features(config, datos)

    available_scale_method = ["normalization", "standardization"]
    X_train, y_train, scale = model_data_preprocessing(config, datos)
    if scale is None:
        raise ValueError(f"Incorrect `scale` method. Only available {available_scale_method}")

    model = model_train(config, X_train, y_train)

    available_model_type = ["random_forest", "logistic"]
    if model is None:
        raise ValueError(f"Incorrect model name. Only available {available_model_type}")

    metrics(config, model, X_train, y_train)

    save_model(current_path, model, scale)


if __name__ == '__main__':
    run_training()
