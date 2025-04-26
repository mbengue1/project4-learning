# Mouhamed Mbengue
# Univeristy Of Rochester
# CSc242 - Project 4
# Learning

#imports 
import numpy as np
import sys
import os
import time

def sigmoid(z):
    """
    compute sigmoid function with numerical stabi
    """
    # clip values to avoid overflow in exponen calcs
    z = np.clip(z, -500, 500)
    # standard sigmoid function 1/(1+e^(-z))
    return 1.0 / (1.0 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    """
    compute cross-entropy loss with numerical stab
    """
    # add small epsilon to avoid taking log of zero
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # binary cross-entropy -[y*log(p) + (1-y)*log(1-p)]
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_accuracy(X, y, weights):
    """
    compute prediction accuracy
    """
    # calculate predicted probabilities using sigmoid function
    y_pred_prob = sigmoid(np.dot(X, weights))
    # convert probabilities to binary predictions using 0.5 thresh
    y_pred = (y_pred_prob >= 0.5).astype(int)
    # calculate percentage of correct predictions
    return np.mean(y_pred == y)

def gradient_descent(X, y, learning_rate, num_epochs, seed=None, dev_data=None, verbose=False):
    """
    implement batch gradient descent for logistic regress
    """
    # set random seed for reproducibility if providd
    if seed is not None:
        np.random.seed(seed)
    
    # initialize weights with random normal values
    num_features = X.shape[1]
    weights = np.random.randn(num_features)
    
    # dict to track training metrics
    metrics = {
        'train_accuracy': [],
        'dev_accuracy': [],
        'loss': []
    }
    
    for epoch in range(num_epochs):
        # forward passcompute predictions
        y_pred = sigmoid(np.dot(X, weights))
        
        # compute gradient of crossentropy loss
        # gradient = X^T * (sigmoid(Xw) - y) / n
        gradient = np.dot(X.T, (y_pred - y)) / len(y)
        
        # update weights using gradient descent rule
        weights = weights - learning_rate * gradient
        
        # track metrics if verbose mode is enabled
        if verbose:
            # calc current loss and accuracy
            loss = cross_entropy_loss(y, y_pred)
            train_accuracy = compute_accuracy(X, y, weights)
            metrics['loss'].append(loss)
            metrics['train_accuracy'].append(train_accuracy)
            
            # if development data provided compute accuracy on it
            if dev_data is not None:
                X_dev, y_dev = dev_data
                dev_accuracy = compute_accuracy(X_dev, y_dev, weights)
                metrics['dev_accuracy'].append(dev_accuracy)
            
            # print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}", 
                      file=sys.stderr)
                if dev_data is not None:
                    print(f"Dev Accuracy: {dev_accuracy:.4f}", file=sys.stderr)
    
    return weights, metrics

def load_data(file_path):
    """
    load data and add bias term
    """
    # load spaceseparated data file
    data = np.loadtxt(file_path)
    # separate features (all columns except last) from labels (last column)
    X = data[:, :-1]  # features
    y = data[:, -1]   # labels
    
    # add bias feature (column of 1's) for intercept term
    X_with_bias = np.column_stack((X, np.ones(X.shape[0])))
    
    return X_with_bias, y

def run_multiple_trials(train_file, dev_file, learning_rates, trials_per_rate, num_epochs):
    """
    run multiple trials for each learning rate and collect results
    helper function for testing
    """
    # init dictionary to store results
    results = {}
    
    for lr in learning_rates:
        # create entry for this learning rate
        results[lr] = {
            'train_accuracy': [],
            'dev_accuracy': [],
            'epoch_train_acc': [[] for _ in range(trials_per_rate)],
            'epoch_dev_acc': [[] for _ in range(trials_per_rate)]
        }
        
        # load data once
        X_train, y_train = load_data(train_file)
        X_dev, y_dev = load_data(dev_file) if dev_file else (None, None)
        dev_data = (X_dev, y_dev) if dev_file else None
        
        for trial in range(trials_per_rate):
            # different seed for each trial
            seed = int(time.time()) + trial * 1000
            np.random.seed(seed)
            
            print(f"Running LR={lr}, Trial {trial+1}/{trials_per_rate}", file=sys.stderr)
            
            # track per-epoch metrics
            train_accs = []
            dev_accs = []
            
            # initialize weights 
            num_features = X_train.shape[1]
            weights = np.random.randn(num_features)
            
            # manual implementation of gradient descent to track per-epoch metrics
            for epoch in range(num_epochs):
                # forward pass
                y_train_pred = sigmoid(np.dot(X_train, weights))
                
                # compute and apply gradient
                gradient = np.dot(X_train.T, (y_train_pred - y_train)) / len(y_train)
                weights = weights - lr * gradient
                
                # record accuracies for this epoch
                train_acc = compute_accuracy(X_train, y_train, weights)
                train_accs.append(train_acc)
                
                if dev_data:
                    dev_acc = compute_accuracy(X_dev, y_dev, weights)
                    dev_accs.append(dev_acc)
            
            # save per epoch accuracies 
            results[lr]['epoch_train_acc'][trial] = train_accs
            if dev_data:
                results[lr]['epoch_dev_acc'][trial] = dev_accs
            
            # store final accuracies for summary statistics
            final_train_acc = train_accs[-1]
            results[lr]['train_accuracy'].append(final_train_acc)
            
            if dev_data:
                final_dev_acc = dev_accs[-1]
                results[lr]['dev_accuracy'].append(final_dev_acc)
                
            # print results
            print(f"LR: {lr}, Trial: {trial+1}, Train Acc: {final_train_acc:.4f}", file=sys.stderr)
            if dev_data:
                print(f"Dev Acc: {final_dev_acc:.4f}", file=sys.stderr)
    
    # print summary statistics for each learning rate
    for lr in learning_rates:
        train_accs = results[lr]['train_accuracy']
        print(f"\nLearning Rate: {lr}", file=sys.stderr)
        print(f"Train accuracy: mean={np.mean(train_accs):.4f}, min={np.min(train_accs):.4f}, max={np.max(train_accs):.4f}", file=sys.stderr)
        
        if results[lr]['dev_accuracy']:
            dev_accs = results[lr]['dev_accuracy']
            print(f"Dev accuracy: mean={np.mean(dev_accs):.4f}, min={np.min(dev_accs):.4f}, max={np.max(dev_accs):.4f}", file=sys.stderr)
    
    return results

def run_systematic_exploration(train_file, num_epochs):
    """
    run the systematic exploration of learning rates
    """
    # learning rates and number of trials 
    learning_rates = [3.0, 1.0, 0.1, 0.01, 0.001]
    trials_per_rate = 5
    
    # look for dev file in same directory as train file
    dev_file = os.path.join(os.path.dirname(train_file), "dev.txt")
    if not os.path.exists(dev_file):
        print(f"Warning: Dev file {dev_file} not found. Will only track training accuracy.", file=sys.stderr)
        dev_file = None
    
    # run the trials and collect results
    print(f"Starting systematic exploration with {len(learning_rates)} learning rates, {trials_per_rate} trials each", file=sys.stderr)
    results = run_multiple_trials(train_file, dev_file, learning_rates, trials_per_rate, num_epochs)
    
    print("\nExploration complete. Results should be analyzed and plotted separately.", file=sys.stderr)
    return results

def main():
    """
    main function to parse arguments and run appropriate mode
    """
    # check commandlargss
    if len(sys.argv) < 4:
        print("Usage: python main.py TRAIN_FILE LEARNING_RATE NUM_EPOCHS [--explore]", file=sys.stderr)
        sys.exit(1)
    
    # parse command line arguments
    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])
    
    # check for explore flag as fourth argument
    explore_mode = False
    if len(sys.argv) > 4 and sys.argv[4] == "--explore":
        explore_mode = True
    
    # run in exploration mode if requested
    if explore_mode:
        run_systematic_exploration(train_file, num_epochs)
        sys.exit(0)
    
    # train a single model
    
    # check for optional environment variables
    verbose = os.environ.get('VERBOSE', '0') == '1'
    
    # get random seed from environment or use current time
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
    
    # train the model and measure time
    start_time = time.time()
    weights, _ = gradient_descent(X_train, y_train, learning_rate, num_epochs, 
                               seed=random_seed, dev_data=dev_data, verbose=verbose)
    
    # print timing information if in verbose mode
    if verbose:
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds", file=sys.stderr)
    
    # print weights (feature weights first, then bias as last coefficient)
    print(" ".join(map(str, weights)))

if __name__ == "__main__":
    main() 