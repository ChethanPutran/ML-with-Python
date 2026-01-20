import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(N=100, noise_std=0.2):
    """
    Generate synthetic data for Bayesian Linear Regression.

    Parameters:
        N (int): Number of data points to generate.
        noise_std (float): Standard deviation of the Gaussian noise.
    Returns:
        X (np.ndarray): Input feature matrix of shape (N, 1).
        t (np.ndarray): Target values of shape (N,).
    """
    # X = np.sort(np.random.uniform(0, 1, N)).reshape(-1, 1)
    X = np.linspace(0, 1, N).reshape(-1, 1)
    h = np.sin(2*np.pi*X).ravel() 
    t = h + np.random.normal(0, noise_std, N)
    return X, t, h

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=2.0, basis_function='gaussian'):
        """
        Initialize the Bayesian Linear Regression model.

        Parameters:
            alpha (float): Precision of the prior distribution.
            beta (float): Precision of the likelihood.
        """
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None
        self.basis_function = basis_function
        self.centers = None
    
    def add_basis_function(self, X, scale=0.1, M=10):
        if self.basis_function == 'gaussian': 
            if self.centers is None:
                # Space 10 centers evenly between 0 and 1
                self.centers = np.linspace(X.min(), X.max(), M).reshape(-1, 1)
            
            # s is the width. 0.1 is perfect for a range of 1.0
            s = scale 
            
            # Calculate the Gaussian activation
            # Formula: exp(- ||x - mu||^2 / (2 * s^2))
            diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
            phi = np.exp(-np.sum(diff**2, axis=2) / (2 * s**2))
            
            # Add a bias column (column of 1s)
            bias = np.ones((X.shape[0], 1))
            return np.concatenate([bias, phi], axis=1)
        else:
            return X
    
    def fit(self, X, t):
        """
        Fit the Bayesian Linear Regression model to the data.
        Parameters:
            X (np.ndarray): Input feature matrix of shape (N, D).
            t (np.ndarray): Target values of shape (N,).
        """
        N, D = X.shape

        phi_X = self.add_basis_function(X)

        # Compute S_N (posterior covariance)
        S_N_inv = self.alpha * np.eye(phi_X.shape[1]) + self.beta * phi_X.T @ phi_X
        self.S_N = np.linalg.inv(S_N_inv)

        # Compute m_N (posterior mean)
        self.m_N = self.beta * self.S_N @ phi_X.T @ t

    def predict(self, X_new):
        """
        Predict the target values for new input data.

        Parameters:
            X_new (np.ndarray): New input feature matrix of shape (M, D).
        Returns:
            np.ndarray: Predicted target values of shape (M,).
        """
        if self.m_N is None or self.S_N is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with training data before prediction.")

        phi_new_X = self.add_basis_function(X_new)

        # Predictive mean
        y_mean = phi_new_X  @ self.m_N

        # Predictive variance
        y_var = 1 / self.beta + np.sum(phi_new_X @ self.S_N * phi_new_X, axis=1)

        return y_mean, y_var

def plot_results(X, t, h, X_new, y_mean, y_var):

    plt.figure(figsize=(10, 6))
    plt.scatter(X, t, color='blue', label='Training data')
    plt.plot(np.linspace(0,1,len(h)), h, color='green', label='True function')
    plt.plot(X_new, y_mean, color='red', label='Predictive mean')
    plt.fill_between(X_new.ravel(), 
                     y_mean - 2 * np.sqrt(y_var), 
                     y_mean + 2 * np.sqrt(y_var), 
                     color='pink', alpha=0.5, label='Predictive variance (±2 std)')
    plt.title('Bayesian Linear Regression')
    plt.xlabel('Input X')
    plt.ylabel('Target t')
    plt.legend()
    plt.show()



def plot_basis_functions(model : BayesianLinearRegression, scale=0.1):
    # Generate a smooth range of X values to plot the curves
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    # Get the basis function activations
    phi = model.add_basis_function(X_plot, scale=scale)
    
    plt.figure(figsize=(10, 4))
    # We skip the first column because it's the Bias (all 1s)
    plt.plot(X_plot, phi[:, 1:]) 
    
    plt.title("Gaussian Basis Functions ($\sigma=0.1$)")
    plt.xlabel("Input X")
    plt.ylabel("Activation")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    NUM_SAMPLES =100
    NOISE_STD = 0.2

    # Hyperparameters
    MODEL_NOISE_VAR = .1**2  # Assume lower noise for a tighter fit
    PRIOR_DIST_VAR = 1     # Larger variance = smaller alpha = more flexible model
    
    X, t, h = generate_synthetic_data(N=NUM_SAMPLES, noise_std=NOISE_STD)
    
    model = BayesianLinearRegression(
        alpha=1/PRIOR_DIST_VAR, 
        beta=1.0/MODEL_NOISE_VAR, 
        basis_function='gaussian'
    )
    
    # 1. Plot the basis functions
    # plot_basis_functions(model, scale=.05)

    NUM_TRAIN = 10
     # Select a subset of data for training
    idx = np.random.choice(np.arange(NUM_SAMPLES), NUM_TRAIN, replace=False)
    X_train = X[idx]
    t_train = t[idx]
    # 2. Fit and predict
    model.fit(X_train, t_train)
    X_new = np.linspace(0, 1, 100).reshape(-1, 1)
    y_mean, y_var = model.predict(X_new)

    print(y_mean.shape, y_var.shape)

    # 3. Plot final regression results
    plot_results(X_train, t_train, h, X_new, y_mean, y_var)