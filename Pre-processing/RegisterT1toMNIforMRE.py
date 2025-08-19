import os
import ants

# Function to apply transformation using ANTs
def AntsApplyTransmat(movingimage, fixedimg, outname, translist):
    ref_data = ants.image_read(fixedimg)      # Load fixed (reference) image
    mov_Data = ants.image_read(movingimage)   # Load moving image
    warped_img = ants.apply_transforms(fixed=ref_data, moving=mov_Data, transformlist=translist)
    ants.image_write(warped_img, outname)     # Save the registered image

# === Main folder paths === Add your path
SubjectsFolder = "/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE"
MREFolder = os.path.join(SubjectsFolder, "MRE_ToT1_202402")
OutputFolder = os.path.join(SubjectsFolder, "MRE_T1toMNI_202402")
TransformFolder = os.path.join(SubjectsFolder, "RegistrationUtils")
MNI_template = os.path.join(SubjectsFolder, "provacancella/MNI152_T1_1mm_brain.nii.gz")

# === List of input parameter maps ===
all_parameters = ['MRE_stiffness_ToT1_202402.nii.gz']

# === Get list of subject folders ===
ListofSubjectNames = [d for d in os.listdir(MREFolder)
                      if os.path.isdir(os.path.join(MREFolder, d)) and ('PD20' in d or 'PD_20' in d)]

# === Main loop through subjects and parameters ===
for subj_path in ListofSubjectNames:
    subj_name = os.path.basename(subj_path)
    
    print(f"\n--- Processing subject: {subj_name} ---")
    
    # Define subject-specific input paths
    subj_mre = os.path.join(MREFolder, subj_name)
    subj_transform = os.path.join(TransformFolder, subj_name, "t1toMNItransmat_202402.mat")  #
    
    if not os.path.exists(subj_transform):
        print(f"Transform file not found: {subj_transform}")
        continue

    for param_file in all_parameters:
        param_path = os.path.join(subj_mre, param_file)
            
        if not os.path.exists(param_path):
            print(f"Missing parameter file: {param_path}")
            continue

        # Create subject-specific output folder
        subj_output_folder = os.path.join(OutputFolder, subj_name)
        os.makedirs(subj_output_folder, exist_ok=True)

        # Construct output filename inside that subject's folder
        param_name_noext = os.path.splitext(os.path.splitext(param_file)[0])[0]  # Remove .nii.gz
        output_path = os.path.join(subj_output_folder, f"{param_name_noext}_t1_to_MNI.nii.gz")

        try:
            # Apply the transformation
            AntsApplyTransmat(movingimage=param_path,
                              fixedimg=MNI_template,
                              outname=output_path,
                              translist=[subj_transform])
            print(f"Transformed: {param_file}")
        except Exception as e:
            print(f"Error in subj {subj_name} - file {param_file}: {e}")