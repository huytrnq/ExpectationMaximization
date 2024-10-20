import numpy as np
import nibabel as nib

from utils import calculate_dice_score
from em_algo import ExpectationMaximization


if __name__ == '__main__':
    # Load the two NIfTI images
    T1_path = 'data/T1.nii'
    T2_path = 'data/T2_FLAIR.nii'
    label_path = 'data/Labels.nii'
    
    ### Load the images from the path
    T1 = nib.load(T1_path)
    T1_np_img = T1.get_fdata()
    T2 = nib.load(T2_path)
    T2_np_img = T2.get_fdata()
    labels = nib.load(label_path)
    labels_np_img = labels.get_fdata()
    
    # Perform Skull Stripping to get the brain mask for WM, GM, and CSF
    ## Isolate the brain voxels using the label
    #### Get the indices of the brain voxels
    brain_voxels_indices = np.where(labels_np_img > 0)
    #### Get the brain voxels from the T1 image using the indices get from the label
    T1_skull_stripped = T1_np_img[brain_voxels_indices]
    T2_skull_stripped = T2_np_img[brain_voxels_indices]
    
    ### Stack the T1 and T2 images to get the multi-modal image
    X = np.vstack((T1_skull_stripped, T2_skull_stripped)).T
    
    # Initialize the Expectation Maximization algorithm
    em = ExpectationMaximization(X, k=3, max_iter=50, type='kmeans')
    alphas, mus, covars, W = em.fit()
    
    # Assign each voxel to the cluster with the highest probability
    voxel_assignments = np.argmax(W, axis=1)
    ### Create a new image with the same shape as the original image with the voxel assignments
    segmented_image = np.zeros_like(T1_np_img)
    ### Assign the voxel assignments to the segmented image
    segmented_image[brain_voxels_indices] = voxel_assignments + 1
    
    ### Manually correct the labels
    segmented_image[segmented_image == 2] = -1
    segmented_image[segmented_image == 3] = 2
    segmented_image[segmented_image == -1] = 3
    
    ### Save the segmented image
    segmented_nii = nib.Nifti1Image(segmented_image, T1.affine)
    nib.save(segmented_nii, 'data/Segmented.nii')
    
    # Calculate the Dice score between the predicted labels and the ground truth
    dice_scores = calculate_dice_score(label_path, 'data/Segmented.nii')    
    print(dice_scores)
    print('Average Dice Score: ', np.mean(list(dice_scores.values())))
    
    
