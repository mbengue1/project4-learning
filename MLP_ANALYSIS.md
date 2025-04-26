# mlp and sgd analysis

## mlp implementation

implemented a multilayer perceptron (mlp) with one hidden layer containing 10 hidden units, all with sigmoid activation functions. the implementation includes:

- forward propagation with sigmoid activation
- backward propagation for gradient computation
- support for both batch and stochastic gradient descent
- metrics tracking for loss and accuracy

## experimental results

### learning rate comparison (mlp vs. logistic regression)

| learning rate | model | final train acc | final dev acc | convergence speed |
|---------------|-------|-----------------|---------------|-------------------|
| 0.1 | logistic regression | 91.95% | 90.22% | moderate |
| 0.1 | mlp | 54.10% | 55.89% | very slow |
| 1.0 | logistic regression | 99.55% | 97.41% | fast |
| 1.0 | mlp (sgd, batch=32) | 100.00% | 100.00% | fast |
| 3.0 | logistic regression | 99.55% | 97.68% | very fast |
| 3.0 | mlp | 99.55% | 97.60% | fast |

### key observations:

1. **learning rate sensitivity**: the mlp is more sensitive to learning rate than logistic regression. with a learning rate of 0.1, the mlp failed to converge within 50 epochs, while logistic regression achieved better accuracy.

2. **performance ceiling**: at optimal learning rates, the mlp can achieve superior performance compared to logistic regression. with sgd and learning rate 1.0, the mlp reached 100% accuracy on both training and development sets.

3. **convergence patterns**: the mlp shows different convergence patterns:
   - with low learning rates (0.1): barely moves from initialization
   - with high learning rates (3.0): converges rapidly, similar to logistic regression
   - with sgd: converges faster and more reliably than batch gradient descent

## sgd analysis

tested sgd with different batch sizes to analyze its impact on:
- convergennce speed
- final accuracy
- computational efficiency

### batch size comparison (learning rate = 1.0, 50 epochs)

| batch size | final train acc | final dev acc | training time | epochs to >99% dev acc |
|------------|-----------------|---------------|---------------|------------------------|
| 8 | 100.00% | 100.00% | 0.27s | ~10 |
| 32 | 100.00% | 100.00% | 0.11s | ~10 |
| full batch | 99.55% | 97.60% | 0.02s | ~30 |

### flop analysis

for mlp with n training examples, d input features, h hidden units, and batch size b:

- forward propagation: O(b × (d × h + h × 1)) flops per batch
- backward propagation: O(b × (h × 1 + d × h)) flops per batch
- total flops per epoch: O((n/b) × b × (d × h + h × 1) × 2) = O(n × (d × h + h) × 2)

observations:
- smaller batch sizes require more parameter updates per epoch
- but each update is computationally cheaper
- wall-clock time is optimized with moderate batch sizes (e.g., 32), balancing:
  - enough parallelism for efficient matrix operations
  - frequent enough updates to improve convergence

### sgd advantages

1. **faster convergence**: sgd with batch size 8 and 32 reached 100% dev accuracy in ~10 epochs, while full batch gradient descent required ~30 epochs to reach 97.6%.

2. **escaping local minima**: the stochastic nature of sgd helps escape poor local minima, leading to better final accuracy.

3. **memory efficiency**: smaller batches require less memory, which is beneficial for larger datasets.

### sgd disadvantages

1. **training time**: smaller batch sizes increase overall training time due to more frequent updates.

2. **convergence noise**: sgd introduces noise in the optimization process, which can make convergence less stable.

3. **hyperparameter sensitivity**: sgd requires tuning of both learning rate and batch size.

## comparison to logistic regression

the mlp shows several differences compared to logistic regression:

1. **model capacity**: the mlp can learn more complex decision boundaries, leading to higher potential accuracy.

2. **parameter count**: the mlp has many more parameters (4 × d + 31 for our implementation with 10 hidden units) compared to logistic regression (d + 1).

3. **optimization difficulty**: the mlp's loss landscape is more complex, making it more sensitive to initialization and learning rate.

4. **performance ceiling**: on this dataset, the mlp with sgd can achieve perfect accuracy (100% on dev), while logistic regression peaks at around 97.7%.

5. **convergence speed**: with appropriate hyperparameters, the mlp can actually converge faster than logistic regression, especially with sgd.

## conclusion

the mlp with sgd demonstrates superior performance compared to logistic regression on this dataset, achieving perfect accuracy on the development set. the key insights are:

1. mlps require careful tuning of hyperparameters, with higher learning rates (1.0-3.0) working surprisingly well.

2. sgd with moderate batch sizes (8-32) offers the best trade-off between convergence speed and computational efficiency.

3. the additional complexity of mlps pays off in terms of final accuracy, but comes at the cost of more hyperparameters to tune.

4. for this specific dataset, the higher capacity of the mlp allows it to capture patterns that logistic regression cannot, resulting in better generalization. 