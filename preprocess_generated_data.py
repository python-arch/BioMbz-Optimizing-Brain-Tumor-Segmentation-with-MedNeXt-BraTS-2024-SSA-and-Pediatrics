import os
import shutil
import json
import random

# Load the JSON data from "/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/brats_ssa_2023_5_fold.json"
with open('/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/brats_ssa_2023_5_fold.json', 'r') as json_file:
    data = json.load(json_file)

def rename_file(subfolder_name, file_name):
    if "seg" in file_name:
        return f"{subfolder_name}-seg.nii.gz"
    elif "t1c" in file_name:
        return f"{subfolder_name}-t1c.nii.gz"
    elif "t1n" in file_name:
        return f"{subfolder_name}-t1n.nii.gz"
    elif "t2f" in file_name:
        return f"{subfolder_name}-t2f.nii.gz"
    elif "t2w" in file_name:
        return f"{subfolder_name}-t2w.nii.gz"
    return file_name

def copy_and_rename_subfolders(src_folder, dest_folder, format_str):
    # Ensure the source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} does not exist.")
        return
    
    # Ensure the destination folder exists or create it
    if not os.path.exists(dest_folder):
        print(f"Destination folder {dest_folder} does not exist.")
        return
    
    # Get the list of subfolders in the source folder
    subfolders = [f for f in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, f))]
    
    # Copy and rename each sample folder
    for idx, subfolder in enumerate(subfolders):
        src_path = os.path.join(src_folder, subfolder)
        new_folder_name = f"{format_str}{idx:03}"
        dest_path = os.path.join(dest_folder, new_folder_name)
        
        # Copy the folder
        os.makedirs(dest_path, exist_ok=True)
        # store the images paths in list
        images_paths = []
        seg_name = ""
        # Rename and copy the files within the samples (00 for example) subfolder
        for index,file_name in enumerate(os.listdir(src_path)):
            full_file_name = os.path.join(src_path, file_name)
            if os.path.isfile(full_file_name):
                new_file_name = rename_file(new_folder_name, file_name)
                if "seg" not in new_file_name:
                    images_paths.append(f"{new_folder_name}/{new_file_name}")
                else:
                    seg_name = F"{new_folder_name}/{new_file_name}"
                shutil.copy(full_file_name, os.path.join(dest_path, new_file_name))
        
         # add the data to the brats json file
        new_entry = {
                        "fold": -1,
                        "image": images_paths,
                        "label": seg_name
                    }
        data['training'].append(new_entry)
        print(f"Copied and renamed {src_path} to {dest_path}")




#usage
src_folder = '/home/abdelrahman.elsayed/Downloads/oriented_270_samples'
dest_folder = '/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'
format_str = 'BraTS-SSA-0000G-'

copy_and_rename_subfolders(src_folder, dest_folder, format_str)

with open('/home/abdelrahman.elsayed/Downloads/BraTS2024_BioMedIAMBZ/dataset/brats_ssa_2023_5_fold.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
