# MNIST Classification with Soft Decision Trees

This repository provides an implementation of two machine learning models for classifying the MNIST dataset: a Soft Decision Tree (SDT) and a Gradient Boosted Soft Decision Tree (GB_SDT). The goal of these models is to demonstrate the effectiveness of decision tree-based approaches in image classification tasks, particularly using the MNIST dataset.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later
- PyTorch
- torchvision
- numpy

These dependencies can be installed using the `requirements.txt` file included in the repository.

### Installation

To install the necessary packages, follow these steps:

1. Clone the repository:

```bash
git clone https://your-repository-url.git
cd your-repository-directory
```

2. Install the relevant python version in .python-version

3. Install the requirements.txt

```bash
pip install -r requirements.txt
```

## Usage

Train the Soft Decision Tree (SDT) model:

```bash
python sdt_train.py --data_dir ./data/mnist --epochs 50 --batch_size 128
```

Train the Gradient Boosted Soft Decision Tree (GB_SDT) model:

```bash
python gb_sdt_train.py --epochs 50 --batch_size 128 --n_trees 4 --depth 5
```

For more options and customization, refer to the help of each script:

```bash
python sdt_train.py --help
python gb_sdt_train.py --help
```

## Frequently Asked Questions

- **Training loss suddenly turns into NAN**
  - **Reason:** Sigmoid function used in internal nodes of SDT can be unstable during the training stage, as its gradient is much close to `0` when the absolute value of input is large.
  - **Solution:** Using a smaller learning rate typically works.

## Experiment Result on MNIST

After training for 40 epochs with `batch_size` 128, the best testing accuracy using a SDT model of depth **5**, **7** are **94.15** and **94.38**, respectively (which is much close to the accuracy reported in raw paper). Related hyper-parameters are available in `main.py`. Better and more stable performance can be achieved by fine-tuning hyper-parameters.

Below are the testing accuracy curve and training loss curve. The testing accuracy of SDT is evaluated after each training epoch.

![MNIST Experiment Result](./mnist_exp.png)

## Package Dependencies

This package is originally developed in `Python 3.11.5`. Following are the name and version of packages used in SDT and GB_SDT. In my practice, it works fine under different versions of Python or PyTorch.

- torch 2.1.2
- torchaudio 2.1.2
- torchvision 0.16.2
