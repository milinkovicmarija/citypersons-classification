# DNN Classification for CityPersons Dataset

This repository contains the code and resources for my master thesis, which focuses on the classification of persons in traffic using the CityPersons dataset. The work includes developing, improving, and evaluating different versions of a Deep Neural Network (DNN) for this classification task.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Versions of the Network](#versions-of-the-network)
- [Dataset](#dataset)

## Introduction

The goal of this thesis is to explore and compare various approaches to person classification in urban traffic scenarios using deep learning. The CityPersons dataset is used as the primary data source. This repository includes the implementation of several network versions, each with different improvements and techniques applied.

## Repository Structure

The repository is organized as follows:

citypersons-classification/
├── configs
│   ├── basic_config.yaml
│   ├── focal_loss_config.yaml
│   ├── improved_config.yaml
│   ├── oversampling_config.yaml
│   ├── synthetic_data_config.yaml
│   └── wce_config.yaml
├── data
│   ├── processed
│   ├── raw
│   └── synthetic
├── src
│   ├── cityscapesScripts
│   └── models
│       ├── init.py
│       ├── basic_network.py
│       ├── focal_loss_network.py
│       ├── improved_network.py
│       ├── oversampling_network.py
│       ├── synthetic_data_network.py
│       └── wce_network.py
│   ├── init.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/milinkovicmarija/citypersons-classification.git
    cd citypersons-classification
    ```

2. **Clone the CityPersonsScripts repository**:
    ```bash
    git clone https://github.com/mcordts/cityscapesScripts.git src/citypersonsscripts
    ```

3. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

1. **Download the dataset**:
    ```bash
    python src/cityscapesScripts/cityscapesscripts/download/downloader.py --d data/raw leftImg8bit_trainvaltest.zip   gtBbox_cityPersons_trainval.zip
    ```

2. **Extract the data**:
    ```bash
    unzip data/raw/leftImg8bit_trainvaltest.zip
    unzip data/raw/gtBbox_cityPersons_trainval.zip
    ```

3. **Preprocess the data**:
    ```bash
    ./src/preprocess_data.sh
    ```

### Training

Train the desired version of the network:

- Basic Network:
    ```bash
    python src/models/basic_network.py --config configs/basic_config.yaml
    ```

- Improved Network:
    ```bash
    python src/models/improved_network.py --config configs/improved_config.yaml
    ```

- Network with Focal Loss:
    ```bash
    python src/models/focal_loss_network.py --config configs/focal_loss_config.yaml
    ```

- Network with Weighted Cross Entropy:
    ```bash
    python src/models/wce_network.py --config configs/wce_config.yaml
    ```

- Network with Oversampling:
    ```bash
    python src/models/oversampling_network.py --config configs/oversampling_config.yaml
    ```

- Network with Synthetic Data:
    ```bash
    python src/models/synthetic_network.py --config configs/synthetic_data_config.yaml
    ```

### Evaluation

Evaluate the trained models using the corresponding evaluation scripts:

```bash
python src/evaluate_model.py --config configs/basic_config.yaml
```

Replace basic_config.yaml with the appropriate configuration file for other network versions.

## Versions of the Network
The repository includes the following versions of the network:

Basic Network: The initial simple network architecture.
Improved Network: Enhanced version with additional layers and optimizations.
Network with Focal Loss Function: Incorporates the focal loss to handle class imbalance.
Network with Weighted Cross Entropy Loss Function: Uses weighted cross entropy to manage class imbalance.
Network with Oversampling: Applies oversampling techniques to balance the dataset.
Network with Synthetic Data: Utilizes synthetic data to augment the training set.

## Dataset
The CityPersons dataset is used for this project. Make sure to download the dataset as outlined in the Usage section.
