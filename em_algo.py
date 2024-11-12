import time

import numpy as np
from sklearn.cluster import KMeans

from utils import plot_gaussians_on_bars
class ExpectationMaximization:
    def __init__(self, data, k, max_iter=100, type='kmeans', stop_criterion=1e-6, plot_step=-1, save_path=None, show_plot=False, W=None):
        """Expectation Maximization algorithm for Gaussian Mixture Models

        Args:
            data (numpy array): input data of shape (N, d) with N samples and d dimensions
            k (int): number of components
            max_iter (int, optional): Iterations. Defaults to 100.
            type (str, optional): random or kmeans to initialize parameters. Defaults to 'kmeans'.
            stop_criterion (float, optional): stop criterion. Defaults to 1e-6.
            plot_step (int, optional): Display the plot of the clusters at step plot_step, -1 to plot at the end. Defaults to -1.
            save_path (str, optional): path to save the plot. Defaults to None.
            show_plot (bool, optional): Display the plot. Defaults to False.
            W (numpy array, optional): membership weights. Defaults to None.
        """
        self.X = data
        self.k = k
        self.max_iter = max_iter
        self.N, self.d = data.shape
        self.type = type
        self.stop_criterion = stop_criterion
        self.previous_log_likelihood = -np.inf
        self.current_log_likelihood = 0
        self.plot_step = plot_step
        self.save_path = save_path
        self.show_plot = show_plot
        self.initialization(W)
        
    def log_likelihood(self):
        """Compute log likelihood of the Gaussian Mixture Models
        Returns:
            float: log likelihood of the Gaussian Mixture Models
        """
        total_log_likelihood = 0
        current_log_likelihood = 0
        for k in range(self.k):
            total_log_likelihood += np.sum(self.alphas[k] * self.gaussian_mixture_models(self.X, self.mus[k], self.covars[k]))
        
        current_log_likelihood += np.log(total_log_likelihood)  
        return current_log_likelihood

        
    def fit(self):
        """Fit the Gaussian Mixture Models

        Returns:
            tuple: alphas, mus, covars, W
        """
        for iteration in range(self.max_iter):
            start = time.time()
            self.expectation()
            self.maximization()
            ### Compute log likelihood
            self.current_log_likelihood = self.log_likelihood()
            end = time.time()
            print('Iteration: ', iteration, ' --- Log Likelihood: ', self.current_log_likelihood, ' --- Time (s): ', end - start)
            ### Plot the clusters
            if iteration % self.plot_step == 0 or iteration == self.max_iter - 1:
                if self.d == 1:
                    plot_gaussians_on_bars(self.X, self.mus, self.covars, iteration, save_path=self.save_path, show=self.show_plot)
                else:
                    plot_gaussians_on_bars(self.X, self.mus, np.diagonal(self.covars, axis1=1, axis2=2), iteration, save_path=self.save_path, show=self.show_plot)  
                    
            ### Check for convergence
            if np.abs(self.current_log_likelihood - self.previous_log_likelihood) < self.stop_criterion:
                break
            else:
                self.previous_log_likelihood = self.current_log_likelihood
        return self.alphas, self.mus, self.covars, self.W
    
    def fit_atlas(self, update_frequency=1, atlas_W=None):
        """Fit the Gaussian Mixture Models with ATLAS integration
        
        Args:
            update_frequency (int, optional): frequency to update the ATLAS. Defaults to 1.
            atlas_W (numpy array, optional): membership weights of the ATLAS. Defaults to None.

        Returns:
            tuple: alphas, mus, covars, W
        """
        for iteration in range(self.max_iter):
            start = time.time()
            if iteration != 0:
                self.expectation()
            self.maximization()
            ### Update the ATLAS weights with the current weights
            if update_frequency and iteration % update_frequency == 0:
                self.W = atlas_W * self.W
                self.W = self.W / np.sum(self.W, axis=1)[:, np.newaxis]
            ### Compute log likelihood
            self.current_log_likelihood = self.log_likelihood()
            end = time.time()
            print('Iteration: ', iteration, ' --- Log Likelihood: ', self.current_log_likelihood, ' --- Time (s): ', end - start)
            ### Plot the clusters
            if iteration % self.plot_step == 0 or iteration == self.max_iter - 1:
                if self.d == 1:
                    plot_gaussians_on_bars(self.X, self.mus, self.covars, iteration, save_path=self.save_path, show=self.show_plot)
                else:
                    plot_gaussians_on_bars(self.X, self.mus, np.diagonal(self.covars, axis1=1, axis2=2), iteration, save_path=self.save_path, show=self.show_plot)  
                    
            ### Check for convergence
            if np.abs(self.current_log_likelihood - self.previous_log_likelihood) < self.stop_criterion:
                break
            else:
                self.previous_log_likelihood = self.current_log_likelihood
        return self.alphas, self.mus, self.covars, self.W
        
        
    def initialization(self, W=None):
        """Initialize parameters of the Gaussian Mixture Models
        Args:
            W (numpy array, optional): membership weights. Defaults to None.
        Raises:
            ValueError: Invalid initialization type
        """
        ### Initialize mixing coefficients
        self.alphas = np.ones(self.k) / self.k
        ### Initialize mixture weights
        self.W = np.zeros((self.N, self.k)) if W is None else W
        #### Initialize mean with kmeans
        if self.type == 'kmeans':
            ### Initialize means
            kmeans = KMeans(n_clusters = self.k, random_state=0).fit(self.X)
            self.mus = kmeans.cluster_centers_
            ### Initialize covariance matrix
            cov_matrix = np.cov(self.X, rowvar=False)  # Covariance matrix of the entire dataset (shape d x d)
            # Initialize K covariance matrices as copies of the dataset's covariance matrix
            self.covars = np.array([cov_matrix for _ in range(self.k)])
        elif self.type == 'random':
            # Randomly initialize means (mus) within the range of the data
            self.mus = np.random.uniform(np.min(self.X), np.max(self.X), size=(self.k, self.d))
            # Randomly initialize covariances (covars)
            if self.d == 1:
                # For 1D, covariances are scalars (variances)
                self.covars = np.random.uniform(0, np.var(self.X), size=self.k)
            else:
                # For d-dimensional data, we need positive semi-definite matrices
                self.covars = np.zeros((self.k, self.d, self.d))
                for i in range(self.k):
                    # Create a random dxd matrix and use dot product to make it positive semi-definite
                    A = np.random.uniform(0, np.var(self.X), size=(self.d, self.d))
                    self.covars[i] = np.dot(A, A.T)  # A * A.T makes it positive semi-definite
                    # Add small diagonal to ensure numerical stability
                    self.covars[i] += np.eye(self.d) * 1e-6
        else:
            raise ValueError('Invalid initialization type')
    
            
    def expectation(self):
        """Expectation step of the Expectation Maximization algorithm
        """
        ### Compute the membership weights
        for k in range(self.k):
            self.W[:, k] = self.alphas[k] * self.gaussian_mixture_models(self.X, self.mus[k], self.covars[k])
        ### Normalize the membership weights
        self.W = self.W / np.sum(self.W, axis=1)[:, np.newaxis]
    
            
    def maximization(self):
        """Maximization step of the Expectation Maximization algorithm
        """

        for k in range(self.k):
            N_k = np.sum(self.W[:, k])
            self.alphas[k] = N_k / self.N
            ### Update means
            ### Transpose X to compute the outer product
            self.mus[k] = (1 / N_k) * self.X.T @ self.W[:, k]
            ### Transpose X to compute the outer product and use element-wise multiplication
            self.covars[k] = (1 / N_k) * (self.X - self.mus[k]).T @ (self.W[:, k][:, np.newaxis] * (self.X - self.mus[k]))
            
            
    def gaussian_mixture_models(self, X, mu, covar):
        """Gaussian Mixture Models of components k

        Args:
            X (numpy array): input data of shape (N, d) with N samples and d dimensions
            mu (numpy array): mean of the Gaussian distribution of components k
            covar (numpy array): covariance of the Gaussian distribution of components k

        Returns:
            numpy array: probability of the Gaussian distribution of components k
        """

        # Regularize covariance matrix
        if self.d == 1:
            covar = np.array([[covar]])  # Ensure 2D structure for 1D case
            covar_inv = 1 / covar
            covar_det = covar
        else:
            covar_inv = np.linalg.inv(covar)  # Inverse of covariance matrix
            covar_det = np.linalg.det(covar)  # Determinant of covariance matrix
        
        # Calculate the Gaussian probability
        diff = X - mu  # Difference between data points and the mean
        numerator = np.exp(-0.5 * np.sum(((diff @ covar_inv) * diff), axis=1))  # Exponent term
        denominator = np.sqrt((2 * np.pi) ** self.d * covar_det)  # Denominator

        p_k = numerator / denominator  # Gaussian probability
        return p_k