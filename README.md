# Multi-Font Letter Recognition Using an MLP Neural Network (9×7 Patterns) with Noise Robustness Testing

This repository presents a Python implementation of **multi-font letter recognition** using a **Multilayer Perceptron (MLP)** neural network. The project evaluates the classification of symbolic letter patterns represented as small **9×7 binary grids**, with additional testing under several types of input noise to assess robustness.

The main goal of the project is to show that even a relatively small fully connected neural network can successfully classify structured symbolic patterns when the input space is compact and well-defined.

---

## Project Overview

The task is to recognize the following seven letters:

- **A**
- **B**
- **C**
- **D**
- **E**
- **J**
- **K**

Each letter is represented in **three different fonts**, producing a total of **21 samples**.  
Every letter pattern is stored as a **9×7 binary image**, then flattened into a vector of length **63** for neural network input.

Pixel encoding:

- **1** = active / bright pixel
- **-1** = inactive / dark pixel

---

## What I Implemented

This repository includes:

- definition of 9×7 symbolic letter patterns for 7 classes across 3 fonts
- conversion of each pattern into a 63-dimensional input vector
- construction of a full dataset with **21 samples**
- a **multiclass MLP classifier** for direct 7-letter recognition
- a **binary MLP classifier** for detecting **A vs non-A**
- evaluation of model performance on the training set
- robustness testing under three different noise models
- comparison of classification behavior under noisy inputs

---

## Method

### Dataset Structure

- **Number of letters:** 7
- **Fonts per letter:** 3
- **Total samples:** 21
- **Pattern size:** 9×7
- **Flattened feature size:** 63

### Multi-Class Model

The multiclass model is designed to classify all 7 letters directly.

Configuration:

- **input size:** 63
- **hidden layer:** 12 neurons
- **activation:** `tanh`
- **solver:** stochastic gradient descent (`sgd`)
- **initial learning rate:** 0.1
- **maximum iterations:** 3000

### Binary Model

A second MLP is trained for:

- **class 1:** letter **A**
- **class 0:** all other letters

Configuration:

- **hidden layer:** 10 neurons
- **activation:** `tanh`
- **solver:** `sgd`
- **initial learning rate:** 0.1
- **maximum iterations:** 3000

---

## Noise Robustness Testing

To evaluate robustness, three types of noise are applied to the input patterns:

1. **Active-to-inactive flip (`1 → -1`)**  
   Simulates losing parts of a letter.

2. **Inactive-to-active flip (`-1 → 1`)**  
   Simulates extra unwanted pixels in the background.

3. **Unknown / removed pixel (`±1 → 0`)**  
   Simulates uncertainty or missing information.

The project evaluates both the multiclass and binary models under these perturbations.

---

## Results

### Training Accuracy

Both models achieved perfect accuracy on the training data:

- **Multi-class training accuracy:** `1.0`
- **Binary training accuracy:** `1.0`

### Noise Evaluation

For the reported test setting, the observed accuracies were:

#### Multi-class model
- **Noise type 1 (`1 → -1`)**: `0.9524`
- **Noise type 2 (`-1 → 1`)**: `1.0`
- **Noise type 3 (`±1 → 0`)**: `1.0`

#### Binary model (`A` vs non-`A`)
- **Noise type 1 (`1 → -1`)**: `1.0`
- **Noise type 2 (`-1 → 1`)**: `1.0`
- **Noise type 3 (`±1 → 0`)**: `1.0`

These results show that the network performs very strongly on this compact symbolic recognition task and remains highly robust under the tested noise conditions.

---

## Why This Project Matters

This project is a useful small-scale example of:

- symbolic pattern recognition
- multiclass neural classification
- binary neural classification
- compact image-like input encoding
- robustness analysis under noisy conditions
- using MLPs for structured low-dimensional vision tasks

It demonstrates that MLP-based classifiers can still be very effective when the input patterns are small, consistent, and information-dense.
