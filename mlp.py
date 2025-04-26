# Mouhamed Mbengue
# Univeristy Of Rochester
# CSc242 - Project 4
# Learning

# imports 
import numpy as np
import sys
import os
import time

def sigmoid(z):
    """
    compute sigmoid function with numerical stability
    """
    # clip values to avoid overflow in exponential calculation
    z = np.clip(z, -500, 500)
    # standard sigmoid function: 1/(1+e^(-z))
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """
    compute derivative of sigmoid function
    """
    s = sigmoid(z)
    return s * (1 - s)

def cross_entropy_loss(y_true, y_pred):
    """
    compute cross-entropy loss with numerical stability
    """
    # add small epsilon to avoid taking log of zero
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_accuracy(y_true, y_pred):
    """
    compute prediction accuracy
    """
    # convert probabilities to binary predictions using 0.5 threshold
    y_pred_binary = (y_pred >= 0.5).astype(int)
    # calculate percentage of correct predictions
    return np.mean(y_pred_binary == y_true)

class MLP:
    """
    multilayer perceptron with one hidden layer and sigmoid activation
    """
    def __init__(self, input_size, hidden_size=10, seed=None):
        """
        initialize mlp with given sizes and random weights
        """
        # set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)
        
        # initialize weights for hidden layer and output layer
        # add bias to both layers
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        
        # store architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def forward(self, X):
        """
        compute forward pass through the network
        """
        # hidden layer
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # output layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def compute_gradients(self, X, y):
        """
        compute gradients using backpropagation
        """
        m = X.shape[0]
        
        # compute forward pass
        y_pred = self.forward(X)
        
        # output layer gradients
        dz2 = y_pred - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # hidden layer gradients
        dz1 = np.dot(dz2, self.w2.T) * sigmoid_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dw1, db1, dw2, db2
    
    def update_parameters(self, dw1, db1, dw2, db2, learning_rate):
        """
        update weights using gradient descent
        """
        # update hidden layer parameters
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        
        # update output layer parameters
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, learning_rate, num_epochs, batch_size=None, dev_data=None, verbose=False):
        """
        train the mlp 
        """
        # track metrics
        metrics = {
            'train_accuracy': [],
            'dev_accuracy': [],
            'loss': []
        }
        
        # number of samples
        m = X.shape[0]
        
        # determine batch size
        if batch_size is None or batch_size >= m:
            batch_size = m  # full batch gradient descent
        
        # training loop
        for epoch in range(num_epochs):
            # shuffle data for stochastic gradient descent
            if batch_size < m:
                indices = np.random.permutation(m)
                X = X[indices]
                y = y[indices]
            
            # initialize loss for this epoch
            epoch_loss = 0
            
            # mini-batch gradient descent
            for i in range(0, m, batch_size):
                # get mini-batch
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # compute gradients
                dw1, db1, dw2, db2 = self.compute_gradients(X_batch, y_batch)
                
                # update parameters
                self.update_parameters(dw1, db1, dw2, db2, learning_rate)
                
                # compute batch loss
                y_pred = self.forward(X_batch)
                batch_loss = cross_entropy_loss(y_batch, y_pred)
                epoch_loss += batch_loss * (X_batch.shape[0] / m)
            
            # track metrics if verbose
            if verbose:
                # compute predictions and metrics on full training set
                y_pred = self.forward(X)
                train_accuracy = compute_accuracy(y, y_pred)
                
                metrics['loss'].append(epoch_loss)
                metrics['train_accuracy'].append(train_accuracy)
                
                if dev_data is not None:
                    X_dev, y_dev = dev_data
                    y_dev_pred = self.forward(X_dev)
                    dev_accuracy = compute_accuracy(y_dev, y_dev_pred)
                    metrics['dev_accuracy'].append(dev_accuracy)
                
                # print progress every 10 epochs
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}", 
                          file=sys.stderr)
                    if dev_data is not None:
                        print(f"Dev Accuracy: {dev_accuracy:.4f}", file=sys.stderr)
        
        return metrics
    
    def predict(self, X):
        """
        predict using the trained mlp
        """
        return self.forward(X)
    
    def get_parameters(self):
        """
        get flattened parameters for output
        """
        # flatten and concatenate all parameters
        w1_flat = self.w1.flatten()
        b1_flat = self.b1.flatten()
        w2_flat = self.w2.flatten()
        b2_flat = self.b2.flatten()
        
        return np.concatenate([w1_flat, b1_flat, w2_flat, b2_flat])

def load_data(file_path):
    """
    load data and prepare for mlp
    """
    # load space-separated data file
    data = np.loadtxt(file_path)
    # separate features from labels
    X = data[:, :-1]  # features
    y = data[:, -1].reshape(-1, 1)  # reshape labels to column vector
    
    return X, y

def main():
    """
    main function to parse arguments and run mlp
    """
    # check cl
    if len(sys.argv) < 4:
        print("Usage: python mlp.py TRAIN_FILE LEARNING_RATE NUM_EPOCHS [--sgd] [--batch-size BATCH_SIZE]", file=sys.stderr)
        sys.exit(1)
    
    # parse command line arguments
    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])
    
    # check for sgd flag and batch size
    use_sgd = False
    batch_size = None
    if "--sgd" in sys.argv:
        use_sgd = True
        try:
            batch_idx = sys.argv.index("--batch-size")
            batch_size = int(sys.argv[batch_idx + 1])
        except (ValueError, IndexError):
            batch_size = 32  # default batch size
    
    # get optional environment variables
    verbose = os.environ.get('VERBOSE', '0') == '1'
    random_seed = os.environ.get('SEED')
    if random_seed:
        random_seed = int(random_seed)
    else:
        random_seed = int(time.time())
    
    # load training data
    try:
        X_train, y_train = load_data(train_file)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # load development data if file exists and verbose mode is on
    dev_data = None
    dev_file = os.path.join(os.path.dirname(train_file), "dev.txt")
    if os.path.exists(dev_file) and verbose:
        try:
            X_dev, y_dev = load_data(dev_file)
            dev_data = (X_dev, y_dev)
        except Exception as e:
            print(f"Error loading dev file: {e}", file=sys.stderr)
    
    # create and train mlp
    input_size = X_train.shape[1]
    mlp = MLP(input_size=input_size, hidden_size=10, seed=random_seed)
    
    # train the model and measure time
    start_time = time.time()
    mlp.train(X_train, y_train, learning_rate, num_epochs, 
              batch_size=batch_size if use_sgd else None,
              dev_data=dev_data, verbose=verbose)
    
    # print timing information if in verbose mode
    if verbose:
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds", file=sys.stderr)
    
    # get and print parameters
    params = mlp.get_parameters()
    print(" ".join(map(str, params)))

if __name__ == "__main__":
    main() 