import numpy as np
from sklearn.cluster import KMeans

class ExpectationMaximization:
    def __init__(self, data, k, max_iter=100, type='kmeans', stop_criterion=1e-6):
        """Expectation Maximization algorithm for Gaussian Mixture Models

        Args:
            data (numpy array): input data of shape (N, d) with N samples and d dimensions
            k (int): number of components
            max_iter (int, optional): Iterations. Defaults to 100.
            type (str, optional): random or kmeans to initialize parameters. Defaults to 'kmeans'.
        """
        self.X = data
        self.k = k
        self.max_iter = max_iter
        self.N, self.d = data.shape
        self.type = type
        self.stop_criterion = stop_criterion
        self.previous_log_likelihood = -np.inf
        self.current_log_likelihood = 0
        self.initialization()
        
    def log_likelihood(self):
        """Compute log likelihood of the Gaussian Mixture Models
        Returns:
            float: log likelihood of the Gaussian Mixture Models
        """
        total_log_likelihood = 0
        current_log_likelihood = 0
        for i in range(self.N):
            for k in range(self.k):
                total_log_likelihood += self.alphas[k] * self.gaussian_mixture_models(self.X[i], self.mus[k], self.covars[k])
            current_log_likelihood += np.log(total_log_likelihood)
        return current_log_likelihood
        
    def fit(self):
        for iteration in range(self.max_iter):
            self.expectation()
            self.maximization()
            ### Compute log likelihood
            self.current_log_likelihood = self.log_likelihood()
            print('Iteration: ', iteration, 'Log Likelihood: ', self.current_log_likelihood)
            if np.abs(self.current_log_likelihood - self.previous_log_likelihood) < self.stop_criterion:
                break
            else:
                self.previous_log_likelihood = self.current_log_likelihood
        return self.alphas, self.mus, self.covars, self.W
        
    def initialization(self):
        """Initialize parameters of the Gaussian Mixture Models

        Raises:
            ValueError: Invalid initialization type
        """
        ### Initialize mixing coefficients
        self.alphas = np.ones(self.k) / self.k
        ### Initialize mixture weights
        self.W = np.zeros((self.N, self.k))
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
            self.mus = np.random.rand(self.k, self.d)
            self.covars = np.array([np.eye(self.d) for _ in range(self.k)])
        else:
            raise ValueError('Invalid initialization type')
    
            
    def expectation(self):
        """Expectation step of the Expectation Maximization algorithm
        """
        ### Compute the membership weights
        for k in range(self.k):
            for i in range(self.N):
                self.W[i, k] = self.alphas[k] * self.gaussian_mixture_models(self.X[i], self.mus[k], self.covars[k])
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
            
            
    def gaussian_mixture_models(self, x, mu, covar):
        """Gaussian Mixture Models of components k

        Args:
            x (float): input data
            mu (float): mean of the Gaussian distribution of components k
            covar (float): covariance of the Gaussian distribution of components k
            d (int): dimension of the data
        Returns:
            float: probability of the Gaussian distribution of components k
        """

        covar_inv = np.linalg.inv(covar)
        covar_det = np.linalg.det(covar)
        
        numerator = np.exp(-0.5 * np.dot(np.dot((x - mu).T, covar_inv), (x - mu)))
        denominator = np.sqrt((2 * np.pi) ** self.d * covar_det)
        # Avoid invalid values in sqrt by ensuring determinant is positive
        p_k = numerator / denominator
        return p_k