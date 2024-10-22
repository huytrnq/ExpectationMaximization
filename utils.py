import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_dice_score(nii_image1_path, nii_image2_path, ignore_background=True):
    """
    Calculate the Dice score between two NIfTI images.
    
    Parameters:
    nii_image1_path (str): Path to the first NIfTI image (e.g., ground truth).
    nii_image2_path (str): Path to the second NIfTI image (e.g., predicted labels).
    ignore_background (bool): Whether to ignore the background label (default: True).
    
    Returns:
    dice_scores (dict): Dictionary of Dice scores for each label present in the images.
    """
    # Load the two NIfTI images
    img1 = nib.load(nii_image1_path)
    img2 = nib.load(nii_image2_path)
    
    # Extract image data as NumPy arrays
    img1_data = img1.get_fdata().astype(np.int32)
    img2_data = img2.get_fdata().astype(np.int32)
    
    # Ensure the two images have the same shape
    if img1_data.shape != img2_data.shape:
        raise ValueError("The two NIfTI images must have the same shape.")
    
    # Calculate the Dice score for each label
    dice_scores = {}
    labels = np.unique(img1_data)  # Find all unique labels in the first image
    
    for label in labels:
        if label == 0 and ignore_background:
            continue  # Skip background (assuming label 0 is background)
        
        # Create binary masks for the current label in both images
        img1_mask = (img1_data == label)
        img2_mask = (img2_data == label)
        
        # Calculate the intersection and union
        intersection = np.sum(img1_mask & img2_mask)
        union = np.sum(img1_mask) + np.sum(img2_mask)
        
        # Calculate Dice score (handle division by zero)
        if union == 0:
            dice_score = 1.0  # Perfect match if both masks are empty
        else:
            dice_score = (2. * intersection) / union
        
        # Store the Dice score for the current label
        dice_scores[label] = dice_score
    
    return dice_scores


def plotting(iteration, data, labels, save_path=None, show=False):
    """Plot the data points and the clusters
    
    Args:
        iteration (int): iteration number
        data (numpy array): input data of shape (N, d) with N samples and d dimensions
        labels (numpy array): labels of the clusters
        save_path (str, optional): path to save the plot. Defaults to None.
        show (bool, optional): whether to display the plot. Defaults to
    """
    plt.figure()
    plt.title('Iteration: ' + str(iteration))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('T1')
    plt.ylabel('T2')
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(os.path.join(save_path, 'iteration_' + str(iteration) + '.png'))
    if show:    
        plt.show()
        

# Function to plot the bar chart with Gaussian overlays
def plot_gaussians_on_bars(X, mus, covars, iteration, save_path=None, show=False):
    """Plot Gaussian distributions over a histogram of the data for multivariate data

    Args:
        X (numpy array): input data, can be multi-dimensional
        labels (numpy array): cluster labels for each data point
        mus (numpy array): means of the Gaussian distributions (one for each cluster)
        covars (numpy array): covariance matrices of the Gaussian distributions (one for each cluster)
        iteration (int): iteration number
    """
    title = f'Iteration: {iteration}'
    # If X is multi-dimensional, take the first feature (column)
    if X.ndim > 1:
        X_1d = X[:, 0]  # Project onto the first dimension (X-axis)
    else:
        X_1d = X

    # Plot histogram of the data
    plt.hist(X_1d, bins=100, density=True, alpha=0.6, color='black', edgecolor='black', linewidth=1.2)

    # Create an array of evenly spaced values over the range of the data
    x = np.linspace(np.min(X_1d), np.max(X_1d), 10000)

    # Plot each Gaussian distribution
    colors = ['red', 'blue', 'green', 'orange']  # Add more colors if needed
    for i in range(len(mus)):
        # Use the mean projected onto the first dimension
        mu = mus[i][0]  # Use the first component of the mean (X-axis)

        # Covariance matrix: project onto the first dimension (covariance between x and x, i.e., variance in the x direction)
        if covars.ndim == 2:
            sigma = np.sqrt(covars[i][0]) # Extract variance for the first dimension (X-axis)
        else:
            sigma = np.sqrt(covars[i])

        # Compute the 1D Gaussian PDF
        y = norm.pdf(x, mu, sigma)

        # Plot the Gaussian curve
        plt.plot(x, y, label=f'Gaussian {i+1}', color=colors[i % len(colors)], linewidth=2.5)

    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(False)
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(os.path.join(save_path, f'iteration_{iteration}.png'))
    if show:
        plt.show()