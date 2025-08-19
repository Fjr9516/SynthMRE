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

print(f"\n--- Clipping at 5000 and normalizing {parameter_filename} ---")

# === Clip and normalize each subject's image ===
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

        # Clip values above 5000
        img_data_clipped = np.clip(img_data, None, 5000)

        # Normalize to [0, 1] range by dividing by 5000
        img_norm = img_data_clipped / 5000.0

        # Save the normalized image
        normalized_filename = parameter_filename.replace('.nii.gz', '_normalizedTo5000.nii.gz')
        normalized_path = os.path.join(InputFolder, subject, normalized_filename)

        nib.save(nib.Nifti1Image(img_norm, affine, header), normalized_path)
        print(f"Saved: {normalized_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")