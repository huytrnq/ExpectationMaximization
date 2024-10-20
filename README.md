# Expectation-Maximization Algorithm

This repository implements the **Expectation-Maximization (EM) algorithm** for Gaussian Mixture Models (GMM). The EM algorithm is used to find maximum likelihood estimates of parameters in probabilistic models when the data contains latent variables.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#data-preparation)
  - [2. Running the EM Algorithm](#running-the-em-algorithm)
- [Code Structure](#code-structure)
- [License](#license)

---

## Overview

The **Expectation-Maximization** algorithm is a general iterative method for finding maximum likelihood estimates in models where some of the data is missing or hidden. In this implementation, the EM algorithm is applied to **Gaussian Mixture Models (GMMs)**, a probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions with unknown parameters.

### Key Concepts:

- **Expectation (E-step)**: Compute the probability that each data point belongs to each Gaussian component based on the current parameters.
- **Maximization (M-step)**: Update the model parameters (means, covariances, and mixing coefficients) based on these probabilities.

The EM algorithm alternates between these two steps until convergence.

![EM Algorithm](data/em.png)

---

## Features

- **Customizable number of components**: Specify how many Gaussian distributions to fit to the data.
- **Log-likelihood tracking**: Monitor the progress of the EM algorithm by observing the log-likelihood values.
- **Covariance regularization**: Handles numerical stability by regularizing covariance matrices.
- **Stopping criteria**: Supports both log-likelihood convergence and maximum iterations as stopping conditions.

---

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/huytrnq/ExpectationMaximization.git
   ```

2. Navigate to the directory:

   ```bash
   cd ExpectationMaximization
   ```

3. Install the necessary dependencies. You can use `pip` or any other package manager of your choice:

   ```bash
   pip install -r requirements.txt
   ```

---


## Usage

### 1. Data Preparation

You need a dataset with `N` data points and `d` dimensions. The data should be stored in a NumPy array `X` of shape `(N, d)`.
In this example, I have 2 MRI images of the brain, and I want to segment them into 3 regions: white matter, gray matter, and cerebrospinal fluid. The two modalities are T1-weighted and T2-weighted images corresponding to d = 2 dimensions.

### 2. Running the EM Algorithm

You can run the EM algorithm by calling the main function in your code. Here’s an example:

```python
em = ExpectationMaximization(X, k=3, max_iter=50, type='kmeans')
alphas, mus, covars, W = em.fit()
```

### Parameters:

- `X`: A NumPy array containing the dataset with shape `(N, d)`.
- `k`: Number of Gaussian components (clusters) corresponding to the number of regions you want to segment the data into.

---

## Code Structure

```bash
.
├── README.md               # Project documentation
├── data                    # Example data for testing
├── requirements.txt        # Dependencies
├── em_algo.py              # Main EM algorithm implementation
├── utils.py                # Helper functions (dice score)
└── main.py                 # Script to run the EM algorithm on example datasets
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.