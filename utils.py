import numpy as np
import nibabel as nib

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