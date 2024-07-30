import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# basically it loads for the data for each sample with four modailities
def load_data(data_dir, example_id, modalities=["t2f", "t1", "t1c", "t2w"]):
    images = []
    for modality in modalities:
        nii_path = os.path.join(data_dir, example_id, f"{example_id}-{modality}.nii.gz") # get the path for each channel
        nii_img = nib.load(nii_path)
        print(nii_img.shape)
        images.append(nii_img.get_fdata()) # get the data
    seg_path = os.path.join(data_dir, example_id, f"{example_id}-seg.nii.gz")
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    return images, seg_data

def plot_slices(images, seg_data, slice_index=None):
    num_modalities = len(images) 
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, num_modalities)
    
    if slice_index is None:
        slice_index = images[0].shape[2] // 2  # Middle slice
    
    for i in range(num_modalities):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i][:, :, slice_index], cmap="gray")
        ax.set_title(f"Modality {i+1}")
        ax.axis("off")
        
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(images[i][:, :, slice_index], cmap="gray")
        ax.imshow(seg_data[:, :, slice_index], cmap="jet", alpha=0.5)
        ax.set_title(f"Modality {i+1} with Segmentation")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# Example usage
data_dir = "/home/abdelrahman.elsayed/Downloads/oriented_270_samples"
example_id = "BraTS-PED-00249-000"
modalities = ["t2f", "t1n", "t1c", "t2w"]

# Load data
images, seg_data = load_data(data_dir, example_id, modalities)

# Visualize slices
plot_slices(images, seg_data)
