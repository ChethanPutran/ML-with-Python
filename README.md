# ML with Python

A collection of machine learning implementations, experiments, tutorials, and coursework developed while learning and exploring **Machine Learning, Deep Learning, Computer Vision, NLP, and related mathematical foundations**.

The repository contains implementations ranging from classical machine learning algorithms to neural networks, probabilistic models, generative models, and sequence models. It also includes selected implementations of algorithms and exercises originally developed while studying machine learning concepts.

---

## Contents

The repository is organized by machine learning concepts and model families.

### Classical Machine Learning

* **Linear Regression**

  * Linear regression implementations and experiments
  * Model fitting and prediction

* **Logistic Regression**

  * Binary classification
  * Sigmoid-based probabilistic classification

* **K-Nearest Neighbors**

  * Distance-based classification

* **Naive Bayes**

  * Probabilistic classification

* **Decision Trees**

  * Decision tree construction and classification
  * Example applications such as music recommendation

* **Support Vector Machines**

  * Linear and nonlinear SVMs
  * Gaussian/RBF kernels
  * Spam classification experiments

* **Classification**

  * Various classification experiments and implementations

---

## Unsupervised Learning

### K-Means

Implementation and experimentation with **K-Means clustering**, including the iterative process of:

1. Initializing cluster centroids
2. Assigning samples to clusters
3. Updating centroids
4. Repeating until convergence

### Gaussian Mixture Models

Experiments with **Gaussian Mixture Models (GMMs)** and probabilistic clustering.

### PCA

Implementation and experiments involving **Principal Component Analysis (PCA)** for dimensionality reduction and feature representation.

### Bayesian Regression

Exploration of **Bayesian approaches to regression**, including probabilistic treatment of model parameters and predictions.

---

## Deep Learning

### Backpropagation

This section contains implementations and exercises covering the fundamentals of neural-network training, including:

* Sigmoid activation
* Logistic regression
* Regularization
* One-vs-all classification
* Neural networks
* Forward propagation
* Backpropagation
* Gradient computation
* Numerical gradient checking
* Neural-network cost functions

Some of these implementations are based on exercises from the **Stanford Machine Learning coursework**.

### Convolutional Neural Networks

The `CNN` directory contains experiments with **Convolutional Neural Networks**, including convolution-based feature extraction and image classification concepts.

### Sequence Models

The `Sequence` directory contains sequence-modeling experiments, including:

* Recurrent Neural Networks (RNNs)
* Sequential data processing

---

## Generative Models

The `DeepGenerativeModels` directory contains experiments and implementations related to **deep generative modeling**.

Topics explored include the idea of learning probability distributions and generating new samples from learned representations.

---

## Computer Vision

The `ComputerVision` directory contains experiments involving computer vision and image-processing concepts.

Topics include:

* Image processing
* Feature extraction
* Visual classification
* Deep-learning-based vision models

---

## Natural Language Processing

The `NLP` directory contains experiments related to **Natural Language Processing**, covering concepts involved in processing and representing textual data.

---

## State Estimation

### Kalman Filters

The `KalmanFilters` directory explores **Kalman filtering and state estimation**, including the use of probabilistic models to estimate hidden system states from noisy observations.

---

## Optimization

The `Optimization` directory contains implementations and experiments related to optimization techniques used in machine learning.

Optimization is explored both as a standalone topic and as a fundamental component of training machine learning models.

---

## Search and Algorithms

The `Search` directory contains implementations of search algorithms and related algorithmic concepts.

These experiments explore techniques for navigating state spaces and solving search problems.

---

## Tutorials

The `Tutorials` directory contains hands-on notebooks for learning popular deep-learning frameworks.

Current tutorials include:

* **PyTorch**
* **TensorFlow**

These notebooks focus on understanding framework fundamentals through practical experimentation.

---

## Technologies

The repository primarily uses:

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **PyTorch**
* **TensorFlow**
* **Jupyter Notebook**

Some coursework and algorithm implementations also use:

* **MATLAB**

---

## Repository Structure

```text
ML-with-Python/
│
├── Backpropagation/
├── BaysianRegression/
├── CNN/
├── Classification/
├── ComputerVision/
├── DeepGenerativeModels/
├── GMM/
├── KMeans/
├── KNN/
├── KalmanFilters/
├── LinearRegression/
├── LogisticRegression/
├── NLP/
├── NaiveBayes/
├── Optimization/
├── PCA/
├── Search/
├── Sequence/
├── SupportVectorMachine/
├── Trees/
└── Tutorials/
```

The repository is organized primarily around **concepts rather than a single software project**, so individual directories can generally be explored independently.

---

## Learning Approach

The goal of this repository is to understand machine learning by combining:

* Mathematical intuition
* Algorithmic implementations
* From-scratch experimentation
* Framework-based implementations
* Practical datasets
* Visualization
* Model training and evaluation

Rather than treating machine learning models as black boxes, the implementations are used to understand the underlying algorithms and their behavior.

---

## Topics Covered

The repository currently spans the following areas:

| Area                   | Topics                                                                        |
| ---------------------- | ----------------------------------------------------------------------------- |
| Supervised Learning    | Linear Regression, Logistic Regression, KNN, Naive Bayes, Decision Trees, SVM |
| Unsupervised Learning  | K-Means, GMM, PCA                                                             |
| Deep Learning          | Neural Networks, Backpropagation, CNNs, RNNs                                  |
| Generative Modeling    | Deep Generative Models                                                        |
| Computer Vision        | Image Processing and Vision Models                                            |
| NLP                    | Natural Language Processing                                                   |
| Probabilistic Modeling | Bayesian Regression, GMMs                                                     |
| State Estimation       | Kalman Filters                                                                |
| Optimization           | Optimization methods used in ML                                               |
| Algorithms             | Search and related algorithms                                                 |
| Frameworks             | PyTorch, TensorFlow                                                           |
| Coursework             | Stanford ML exercises and related implementations                             |

---

## Purpose

This repository serves as a **hands-on machine learning learning archive** and reference for revisiting algorithms, implementations, experiments, and concepts encountered while studying ML.

It is also intended to evolve over time as new topics, implementations, and experiments are added.

---

## Status

This is an evolving repository. Some directories contain complete implementations and experiments, while others are primarily learning exercises or coursework.

The code should therefore be viewed primarily as **educational and experimental material**, rather than as production-ready machine learning software.
