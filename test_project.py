#!/usr/bin/env python3
import subprocess
import time
import numpy as np
import os
import sys
import unittest

class TestProject(unittest.TestCase):
    def setUp(self):
        """Create test data files before each test"""
        # Create training data
        self.train_data = np.array([
            [0.52, 0.17, -1.23, 0.45, 0.88, 1],
            [-0.25, 0.99, 0.33, -0.12, 0.05, 0],
            [0.77, -0.56, 1.34, 0.60, -0.45, 1],
            [-0.12, 0.45, -0.78, 0.23, 0.67, 0],
            [1.23, -0.89, 0.34, -0.56, 0.12, 1]
        ])
        np.savetxt('test_train.txt', self.train_data)
        
        # Create development data with same format (5 features + 1 label)
        self.dev_data = np.array([
            [0.34, 0.56, -0.78, 0.90, 0.12, 1],
            [-0.45, 0.67, 0.89, -0.23, 0.45, 0],
            [0.56, -0.78, 0.90, 0.12, -0.34, 1]
        ])
        np.savetxt('dev.txt', self.dev_data)

    def tearDown(self):
        """Clean up test files after each test"""
        for file in ['test_train.txt', 'dev.txt']:
            if os.path.exists(file):
                os.remove(file)

    def test_basic_logistic_regression(self):
        """Test basic logistic regression functionality"""
        start_time = time.time()
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', '100'],
                              capture_output=True, text=True)
        end_time = time.time()
        
        # Check execution time
        self.assertLess(end_time - start_time, 30, "Execution time exceeded 30 seconds")
        
        # Check output format
        params = result.stdout.strip().split()
        self.assertEqual(len(params), 6, "Output should have 6 parameters (5 weights + 1 bias)")
        
        # Check all parameters are valid numbers
        for param in params:
            self.assertTrue(self.is_valid_number(param), "Parameters should be valid numbers")

    def test_mlp_implementation(self):
        """Test MLP implementation"""
        start_time = time.time()
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', '100', '--mlp'],
                              capture_output=True, text=True)
        end_time = time.time()
        
        # Check execution time
        self.assertLess(end_time - start_time, 30, "MLP execution time exceeded 30 seconds")
        
        # Check output format
        params = result.stdout.strip().split()
        self.assertGreater(len(params), 6, "MLP should have more parameters than logistic regression")

    def test_sgd_implementation(self):
        """Test SGD implementation"""
        start_time = time.time()
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', '100', '--sgd', '--batch-size', '2'],
                              capture_output=True, text=True)
        end_time = time.time()
        
        # Check execution time
        self.assertLess(end_time - start_time, 30, "SGD execution time exceeded 30 seconds")
        
        # Check output format
        params = result.stdout.strip().split()
        self.assertEqual(len(params), 6, "Output should have 6 parameters (5 weights + 1 bias)")

    def test_mlp_with_sgd(self):
        """Test MLP with SGD"""
        start_time = time.time()
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', '100', '--mlp', '--sgd', '--batch-size', '2'],
                              capture_output=True, text=True)
        end_time = time.time()
        
        # Check execution time
        self.assertLess(end_time - start_time, 30, "MLP with SGD execution time exceeded 30 seconds")
        
        # Check output format
        params = result.stdout.strip().split()
        self.assertGreater(len(params), 6, "MLP should have more parameters than logistic regression")

    def test_verbose_output(self):
        """Test verbose output with development data"""
        # Set environment variable for the subprocess
        env = os.environ.copy()
        env['VERBOSE'] = '1'
        
        # Run with both training and development data
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', '100'],
                              capture_output=True, text=True, env=env)
        
        # Print stderr for debugging
        print("Stderr output:", result.stderr)
        
        # Check that verbose output contains expected information
        self.assertIn('Epoch', result.stderr, "Verbose output should show epoch information")
        self.assertIn('Loss', result.stderr, "Verbose output should show loss information")
        self.assertIn('Train Accuracy', result.stderr, "Verbose output should show training accuracy")
        self.assertIn('Dev Accuracy', result.stderr, "Verbose output should show development accuracy")

    def test_exploration_mode(self):
        """Test exploration mode"""
        start_time = time.time()
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', '100', '--explore'],
                              capture_output=True, text=True)
        end_time = time.time()
        
        # Check execution time
        self.assertLess(end_time - start_time, 30, "Exploration mode execution time exceeded 30 seconds")
        
        # Check that exploration output contains expected information
        self.assertIn('Learning Rate', result.stderr, "Exploration output should show learning rate information")
        self.assertIn('Train accuracy', result.stderr, "Exploration output should show training accuracy")
        self.assertIn('Dev accuracy', result.stderr, "Exploration output should show development accuracy")

    def test_invalid_arguments(self):
        """Test handling of invalid arguments"""
        # Test missing arguments
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1'],
                              capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Should fail with missing arguments")
        
        # Test invalid learning rate
        result = subprocess.run(['python', 'main.py', 'test_train.txt', 'invalid', '100'],
                              capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid learning rate")
        
        # Test invalid number of epochs
        result = subprocess.run(['python', 'main.py', 'test_train.txt', '0.1', 'invalid'],
                              capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid number of epochs")

    def test_performance_with_large_dataset(self):
        """Test performance with a larger dataset"""
        # Create larger training data (1000 samples)
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = (np.random.rand(n_samples) > 0.5).astype(int)
        large_data = np.column_stack((X, y))
        np.savetxt('large_train.txt', large_data)
        
        try:
            start_time = time.time()
            result = subprocess.run(['python', 'main.py', 'large_train.txt', '0.1', '100'],
                                  capture_output=True, text=True)
            end_time = time.time()
            
            # Check execution time
            self.assertLess(end_time - start_time, 30, "Execution time exceeded 30 seconds")
            
            # Check output format
            params = result.stdout.strip().split()
            self.assertEqual(len(params), 6, "Output should have 6 parameters (5 weights + 1 bias)")
            
            # Check all parameters are valid numbers
            for param in params:
                self.assertTrue(self.is_valid_number(param), "Parameters should be valid numbers")
        finally:
            # Clean up
            if os.path.exists('large_train.txt'):
                os.remove('large_train.txt')

    def is_valid_number(self, s):
        """Helper function to check if a string is a valid number"""
        try:
            float(s)
            return True
        except ValueError:
            return False

if __name__ == '__main__':
    unittest.main(verbosity=2) 