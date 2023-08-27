# Tutorial Hydra package to use in a Machine Learning Project

This repo consist of a basic guide to use Hydra package to management configuration files in a ML project

The Machine Learning model was built using scikit-learn.

* Random Forest or Logistic Regression
* Min-Max Scaler or Standard Scaler

Each of these components are developed within the project in an offline setting inside `/model`. 
The scikit-learn model scaler process will still be needed in a production
so they can be serialized via python's pickle functionality and stored within the `/model` folder.


## Installation

Create a conda virtual environment in the project directory.

```
conda create -n hydra_env python=3.9
```

Activate the virtual environment.
```
conda activate hydra_env
```

While in the virtual environment, install required dependencies from `requirements.txt`.

```
pip install -r requirements.txt
```

The application may then be terminated with the following commands.

```
ctrl - c
```

## Project Structure 

```
├── config
├── data
├── model
│   ├── model.joblib
│   └── scale.joblib
├── src
│   ├── load_yaml.py
│   ├── train_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```
