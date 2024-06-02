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
diplomski_rad
├── README.md
├── LICENSE
├── .gitignore
├── data
│ ├── raw
│ ├── processed
│ └── synthetic
├── notebooks
│ ├── exploration.ipynb
│ └── visualization.ipynb
├── src
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── models
│ │ ├── init.py
│ │ ├── basic_network.py
│ │ ├── improved_network.py
│ │ ├── network_focal_loss.py
│ │ ├── network_weighted_ce.py
│ │ ├── network_oversampling.py
│ │ └── network_synthetic_data.py
│ ├── training
│ │ ├── init.py
│ │ ├── train_basic.py
│ │ ├── train_improved.py
│ │ ├── train_focal_loss.py
│ │ ├── train_weighted_ce.py
│ │ ├── train_oversampling.py
│ │ └── train_synthetic.py
│ └── evaluation
│ ├── init.py
│ ├── evaluate_basic.py
│ ├── evaluate_improved.py
│ ├── evaluate_focal_loss.py
│ ├── evaluate_weighted_ce.py
│ ├── evaluate_oversampling.py
│ └── evaluate_synthetic.py
├── configs
│ ├── basic_config.yaml
│ ├── improved_config.yaml
│ ├── focal_loss_config.yaml
│ ├── weighted_ce_config.yaml
│ ├── oversampling_config.yaml
│ └── synthetic_data_config.yaml
├── requirements.txt
└── scripts
├── download_data.sh
├── preprocess_data.sh
└── generate_synthetic_data.py


## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd diplomski_rad
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

1. **Download the dataset**:
    ```bash
    ./scripts/download_data.sh
    ```

2. **Preprocess the data**:
    ```bash
    ./scripts/preprocess_data.sh
    ```

3. **Generate synthetic data** (if needed):
    ```bash
    python scripts/generate_synthetic_data.py
    ```

### Training

Train the desired version of the network:

- Basic Network:
    ```bash
    python src/training/train_basic.py --config configs/basic_config.yaml
    ```

- Improved Network:
    ```bash
    python src/training/train_improved.py --config configs/improved_config.yaml
    ```

- Network with Focal Loss:
    ```bash
    python src/training/train_focal_loss.py --config configs/focal_loss_config.yaml
    ```

- Network with Weighted Cross Entropy:
    ```bash
    python src/training/train_weighted_ce.py --config configs/weighted_ce_config.yaml
    ```

- Network with Oversampling:
    ```bash
    python src/training/train_oversampling.py --config configs/oversampling_config.yaml
    ```

- Network with Synthetic Data:
    ```bash
    python src/training/train_synthetic.py --config configs/synthetic_data_config.yaml
    ```

### Evaluation

Evaluate the trained models using the corresponding evaluation scripts:

```bash
python src/evaluation/evaluate_basic.py --config configs/basic_config.yaml
```

Replace evaluate_basic.py and basic_config.yaml with the appropriate evaluation script and configuration file for other network versions.

## Versions of the Network
The repository includes the following versions of the network:

Basic Network: The initial simple network architecture.
Improved Network: Enhanced version with additional layers or optimizations.
Network with Focal Loss Function: Incorporates the focal loss to handle class imbalance.
Network with Weighted Cross Entropy Loss Function: Uses weighted cross entropy to manage class imbalance.
Network with Oversampling: Applies oversampling techniques to balance the dataset.
Network with Synthetic Data: Utilizes synthetic data to augment the training set.

## Dataset
The CityPersons dataset is used for this project. Make sure to download the dataset as outlined in the Usage section.
