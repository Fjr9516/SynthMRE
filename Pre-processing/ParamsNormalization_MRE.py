import os
import nibabel as nib
import numpy as np

# === Define the main input folder ===
SubjectsFolder = "/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE" #choose the right path
InputFolder = os.path.join(SubjectsFolder, "MRE_T1toMNI_202402")

# === Define the parameter map filename to normalize ===
parameter_filename = 'MRE_stiffness_ToT1_202402_t1_to_MNI.nii.gz'

# === Get list of subject directories containing 'PD20' or 'PD_20' ===
ListOfSubjectNames = [
    d for d in os.listdir(InputFolder)
    if os.path.isdir(os.path.join(InputFolder, d)) and ('PD20' in d or 'PD_20' in d)
]

print(f"\n--- Processing parameter: {parameter_filename} ---")

# === Initialize global min and max ===
global_min = np.inf
global_max = -np.inf

# === Step 1: Compute global min and max across all subjects ===
for subject in ListOfSubjectNames:
    file_path = os.path.join(InputFolder, subject, parameter_filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        img_data = nib.load(file_path).get_fdata()
        img_min = np.min(img_data)
        img_max = np.max(img_data)
        global_min = min(global_min, img_min)
        global_max = max(global_max, img_max)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print(f"Global min: {global_min}, max: {global_max} for {parameter_filename}")

# === Step 2: Normalize each subject's image using global min and max ===
for subject in ListOfSubjectNames:
    file_path = os.path.join(InputFolder, subject, parameter_filename)
    if not os.path.exists(file_path):
        print(f"Skipping missing file: {file_path}")
        continue

    try:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        affine = img.affine
        header = img.header

        # Avoid division by zero
        if global_max > global_min:
            img_norm = (img_data - global_min) / (global_max - global_min)
        else:
            img_norm = np.zeros_like(img_data)

        # Save the normalized image
        normalized_filename = parameter_filename.replace('.nii.gz', '_normalized.nii.gz')
        normalized_path = os.path.join(InputFolder, subject, normalized_filename)

        nib.save(nib.Nifti1Image(img_norm, affine, header), normalized_path)
        print(f"Saved: {normalized_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
