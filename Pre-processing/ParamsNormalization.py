import os
import nibabel as nib
import numpy as np

# === Define main input/output folder ===
a = "/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE" # choose the right path
SubjectsFolder = os.path.join(a, "InputChannels")

# === List of parameter maps to normalize ===
all_parameters = [
    'dtd_covariance_MD_to_t1_antstrans_t1_to_MNI.nii.gz',
    'dtd_covariance_C_mu_to_t1_antstrans_t1_to_MNI.nii.gz',
    'dtd_covariance_FA_to_t1_antstrans_t1_to_MNI.nii.gz',
    'dtd_covariance_V_MD_to_t1_antstrans_t1_to_MNI.nii.gz',
    'dtd_codivide_md_t_to_t1_antstrans_t1_to_MNI.nii.gz',
    'dtd_codivide_v_at_to_t1_antstrans_t1_to_MNI.nii.gz',
    'dtd_codivide_v_fw_to_t1_antstrans_t1_to_MNI.nii.gz',
    't2_to_t1_t1_to_MNI.nii.gz',
    'T1_t1_to_MNI.nii.gz'
]

# === Get list of subjects ===
ListOfSubjectNames = [
    os.path.join(SubjectsFolder, d)
    for d in os.listdir(SubjectsFolder)
    if os.path.isdir(os.path.join(SubjectsFolder, d)) 
       and ('PD20' in d or 'PD_20' in d) 
       and 'aug' not in d
]
print(ListOfSubjectNames)

for filename in all_parameters:
    print(f"\n--- Processing parameter: {filename} ---")
    
    # === Reset min and max for this parameter ===
    maxi = -np.inf
    minu = np.inf

    # === Step 1: compute global min and max for this parameter ===
    for subject in ListOfSubjectNames:
        file_path = os.path.join(SubjectsFolder, subject, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            img_nii = nib.load(file_path)
            img_data = img_nii.get_fdata()

            # Calcolo min e max
            img_min = np.min(img_data)
            img_max = np.max(img_data)
            minu = min(minu, img_min)
            maxi = max(maxi, img_max)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Global min: {minu}, max: {maxi} for {filename}")


        # === Step 2: normalize all subject images using that min/max ===
    for subject in ListOfSubjectNames:
        file_path = os.path.join(SubjectsFolder, subject, filename)
        if not os.path.exists(file_path):
            continue

        try:
            img = nib.load(file_path)
            img_data = img.get_fdata()
            affine = img.affine
            header = img.header

            if maxi > minu:
                img_norm = (img_data - minu) / (maxi - minu)
            else:
                img_norm = np.zeros_like(img_data)

            normalized_filename = filename.replace('.nii.gz', '_normalizedxparam.nii.gz')
            normalized_path = os.path.join(SubjectsFolder, subject, normalized_filename)

            nib.save(nib.Nifti1Image(img_norm, affine, header), normalized_path)
            print(f"Saved: {normalized_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")