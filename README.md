# LSTM Temperature Prediction

This repository contains the implementation of a Long Short-Term Memory (LSTM) neural network for predicting monthly average temperature in Romania using historical climate data.

The project includes two approaches:

* **Manual LSTM implementation from scratch in Python**
* **Comparison model implemented using PyTorch**

## Project Overview

The objective of this project is to forecast monthly average temperature for Romania using historical time-series data collected between **1753 and 2024**.

The manually implemented model reproduces the internal logic of an LSTM architecture, including:

* sequence generation
* gate computation
* hidden state update
* prediction generation
* error evaluation

A second implementation based on **PyTorch** is included for performance comparison.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Tkinter
* PyTorch

## Features

* historical temperature preprocessing
* Min-Max normalization
* sequence creation (12 months window)
* LSTM training
* prediction for 2025
* graphical visualization of results
* RMSE / MAE evaluation

## Publication

Research paper published on Zenodo:

https://zenodo.org/records/19655401

## Files

* `main.py` → manual LSTM implementation + Tkinter interface
* `mainTs.py` → PyTorch implementation
* dataset file → historical monthly temperatures

## Author

Carlos Moldovan
Computer Science Graduate
