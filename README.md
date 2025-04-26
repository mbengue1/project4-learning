# Mouhamed Mbengue
# Univeristy Of Rochester
CSC 242 - Project 4
Learnign

This project implements and analyzes the performance of:
1. Logistic Regression with batch gradient descent
2. Multilayer Perceptron (MLP) with one hidden layer
3. Stochastic Gradient Descent (SGD) optimization

## Project Structure

- `mlp.py`: Implementation of the MLP class with configurable hidden layer size
- `main.py`: Contains implementations for logistic regression and systematic exploration of learning rates
- `plot_results.py`: Functions for visualizing training results and creating learning curves
- `MLP_ANALYSIS.md`: Detailed analysis of MLP and SGD performance compared to logistic regression
- `results.json`: Saved experimental results
- `learning_curves.png`: Visualization of learning curves for different models

## Implementation Details

### Logistic Regression
- Implemented with batch gradient descent
- Supports tracking of training and development accuracy
- Learning rate exploration (0.001 to 3.0)

### Multilayer Perceptron (MLP)
- One hidden layer with configurable size (default: 10 neurons)
- Sigmoid activation functions
- Support for both batch and mini-batch (SGD) gradient descent
- Complete forward and backward propagation implementation

### Optimization Analysis
- Comparison of convergence rates across different learning rates
- Analysis of SGD with various batch sizes (8, 32, full batch)
- FLOP analysis and computational efficiency considerations


## Results Summary

The MLP with SGD achieves superior performance compared to logistic regression on the tested dataset:
- MLP with SGD (batch size 32, learning rate 1.0): 100% accuracy on development set
- Logistic Regression (learning rate 3.0): 97.68% accuracy on development set

For a detailed analysis, see the [MLP_ANALYSIS.md](MLP_ANALYSIS.md) file. # project4-learning
