

# Neural Forge Hub

This repository contains projects completed during the "Introduction to Deep Learning" course at the Technical University of Munich (TUM).

- [Neural Forge Hub](#neural-forge-hub)
  - [1. DataLoader](#1-dataloader)
  - [2. LR-with-Solver](#2-lr-with-solver)
  - [3. NN-Cifar 10](#3-nn-cifar-10)
  - [4. HyperTuneCIFAR10](#4-hypertunecifar10)
  - [5. I2Pytorch](#5-i2pytorch)
  - [6. Autoencoder](#6-autoencoder)
  - [7. FacialKeypointDetection](#7-facialkeypointdetection)


## 1. DataLoader

in the realm of machine learning, additional data preparation steps are often required before embarking on model training.

A pivotal component for such data preparation is the DataLoader. This class proves essential by encapsulating a dataset, enabling us to load small subsets of the dataset iteratively. Instead of loading each sample individually, we can now efficiently work with mini-batches, a term frequently used in machine learning that will play a significant role in upcoming lectures.

In this notebook, my focus will be on implementing a custom DataLoader. This DataLoader will prove instrumental in loading mini-batches from the dataset I previously implemented, enhancing the efficiency of our machine learning workflow.

## 2. LR-with-Solver

LR-with-Solver showcases the implementation of a linear regression model using a custom solver. This project delves into the fundamentals of linear regression and the optimization process involved in solving regression problems.

## 3. NN-Cifar 10

In the previous project phase, we explored binary classification, focusing on common steps and the pivotal role of the "solver" in a logistic regression setup.

In this phase, our goal is to implement a neural network. We'll create self-contained building blocks for building complex models. Specifically, we'll implement the forward and backward passes from scratch, simplifying the process.

## 4. HyperTuneCIFAR10

In the project, we embark on the task of hyperparameter tuning for our model, employing a thoughtful approach encompassing various methods. The tuning strategies include randomized search, grid search, and a nuanced intuition-driven exploration. This deliberate selection of methods aims to optimize our model's performance through a comprehensive exploration of the hyperparameter space.

## 5. I2Pytorch

I2Pytorch is a project dedicated to implementing deep learning models using the PyTorch framework. It covers various aspects of PyTorch usage, including model construction, training, and evaluation.

## 6. Autoencoder

In the initial steps, I downloaded the MNIST dataset, consisting of 60,000 handwritten digit images. To overcome the challenge of labeling such a vast dataset, I strategically labeled a subset of 300 images. Among these, 100 were assigned for training, 100 for validation, and the remaining 100 for testing, posing a unique challenge due to the limited labeled samples.

Throughout the process, I applied various data transformations, including random flipping and rotation, to augment the labeled dataset. However, the true breakthrough came from the fundamental working principle of the autoencoder. The key question was whether I could represent the unlabeled data effectively in a smaller latent space. This critical aspect propelled me to secure the **1st place out of 600** participants on the leaderboard

## 7. FacialKeypointDetection

The projects in this lecture are divided into two main parts. The first part involves reinventing the wheel by implementing essential methods from scratch, while the second part introduces the use of existing libraries that already have these methods implemented. We have now transitioned to the second stage, where we explore more complex network architectures.

With the recent introduction of Convolutional Neural Networks (CNNs), we are equipped with a powerful tool that we'll delve into in this exercise. Your task for this week is to build a Convolutional Neural Network for facial keypoint detection.

Facial keypoints, also known as facial landmarks, are the small magenta dots visible on each face in the images above. These keypoints signify crucial areas of the face, including the eyes, corners of the mouth, nose, etc. They play a vital role in various computer vision tasks such as face filters, emotion recognition, pose recognition, and more. Let's dive into exploring the capabilities of CNNs for this task!
