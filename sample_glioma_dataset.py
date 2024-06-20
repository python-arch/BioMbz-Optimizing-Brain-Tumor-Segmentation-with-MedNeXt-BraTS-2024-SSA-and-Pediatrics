import os
import shutil
import random
import json

# Define the paths
source_dir = '/home/abdelrahman.elsayed/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
destination_dir = '/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'
json_file_path = '/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/brats_ssa_2023_5_fold.json'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


with open('/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/brats_ssa_2023_5_fold.json', 'r') as json_file:
    data = json.load(json_file)

# List all subfolders in the source directory
subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]

# Randomly sample 60 subfolders
sampled_subfolders = random.sample(subfolders, 60)

# Copy the sampled subfolders and create new entries
for subfolder in sampled_subfolders:
    # Get the base name of the subfolder
    base_name = os.path.basename(subfolder)
    
    # Define the destination path for this subfolder
    dest_path = os.path.join(destination_dir, base_name)
    
    # Copy the entire subfolder to the destination directory
    shutil.copytree(subfolder, dest_path)
    
    # Get a list of files in the subfolder
    files = os.listdir(subfolder)
    
    # Separate image files and the label file
    images_paths = [os.path.join(base_name, f) for f in files if 'seg' not in f]
    seg_name = [os.path.join(base_name, f) for f in files if 'seg' in f][0]
    
    # Create the new entry
    new_entry = {
        "fold": -1,
        "image": images_paths,
        "label": seg_name
    }
    
    # Add the new entry to the list
    data['training'].append(new_entry)


# Save the updated data back to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Sampled data points have been copied and JSON file has been updated successfully.")
