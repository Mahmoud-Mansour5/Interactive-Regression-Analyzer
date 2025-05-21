import numpy as np

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target values
        """
        # Convert inputs to numpy arrays and ensure float64 type
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        
        # Initialize best parameters (for early stopping)
        best_weights = self.weights.copy()
        best_bias = self.bias
        best_cost = float('inf')
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            try:
                # Forward pass (make predictions)
                y_predicted = self._predict(X)
                
                # Calculate gradients with numerical stability
                error = y_predicted - y
                dw = (1/n_samples) * np.dot(X.T, error)
                db = (1/n_samples) * np.sum(error)
                
                # Clip gradients to prevent explosion
                dw = np.clip(dw, -1e10, 1e10)
                db = np.clip(db, -1e10, 1e10)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Calculate cost
                current_cost = self._compute_cost(y_predicted, y)
                
                # Store cost history
                self.cost_history.append(current_cost)
                
                # Update best parameters if current cost is better
                if current_cost < best_cost and not np.isnan(current_cost):
                    best_cost = current_cost
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                
                # Early stopping if cost is not improving
                if len(self.cost_history) > 10:
                    if np.mean(np.diff(self.cost_history[-10:])) > 0:
                        break
                
            except (RuntimeWarning, RuntimeError) as e:
                print(f"Warning at iteration {iteration}: {str(e)}")
                break
        
        # Use best parameters found
        self.weights = best_weights
        self.bias = best_bias

    def _predict(self, X):
        """
        Make predictions using the current weights and bias.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Predict using the linear regression model.
        
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        y_pred : numpy array of shape (n_samples,)
            Predicted values
        """
        try:
            predictions = self._predict(X)
            # Clip predictions to prevent extreme values
            return np.clip(predictions, np.min(predictions[~np.isinf(predictions)]), 
                         np.max(predictions[~np.isinf(predictions)]))
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return np.zeros(X.shape[0])

    def _compute_cost(self, y_predicted, y):
        """
        Compute the Mean Squared Error cost function with numerical stability.
        """
        try:
            n_samples = len(y)
            errors = y_predicted - y
            # Clip errors to prevent overflow
            errors = np.clip(errors, -1e10, 1e10)
            cost = (1/(2*n_samples)) * np.sum(errors ** 2)
            return cost if not np.isnan(cost) else float('inf')
        except Exception as e:
            print(f"Error in cost computation: {str(e)}")
            return float('inf')

    def get_params(self):
        """
        Get the model parameters.
        
        Returns:
        dict : Dictionary containing the weights and bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'cost_history': self.cost_history
        }

    def score(self, X, y):
        """
        Calculate the R² score (coefficient of determination).
        
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Test samples
        y : numpy array of shape (n_samples,)
            True values
        
        Returns:
        score : float
            R² score
        """
        try:
            y_pred = self.predict(X)
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            return r2 if not np.isnan(r2) else 0.0
        except Exception as e:
            print(f"Error in score computation: {str(e)}")
            return 0.0 