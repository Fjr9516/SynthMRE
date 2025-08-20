import os
import numpy as np
import torch
import random

def reset_seed_number(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

# Set and apply a fixed seed
seed = 42
reset_seed_number(seed)
g = torch.Generator()
g.manual_seed(seed)

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import skimage.transform
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from monai.networks.nets import UNet
import matplotlib.pyplot as plt
import nibabel as nib
import time

from torch.utils.tensorboard import SummaryWriter
import re
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import statsmodels.formula.api as smf
import pandas as pd
from copy import deepcopy
import math

from numpy import asarray, float32
from torch import from_numpy, sum as tsum, stack, cat, float32 as tfloat32
from torch.nn import Module, Conv3d, L1Loss
from torch.nn.functional import pad as tpad

# Gradient Magnitude Edge Loss (GMELoss)
def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    '''
    Returns 3D Sobel kernels Sx, Sy, Sz, & diagonal kernels for edge detection.

    Parameters
    ----------
    n1 : int, optional
        Kernel value 1 (default 1).
    n2 : int, optional
        Kernel value 2 (default 2).
    n3 : int, optional
        Kernel value 3 (default 2).

    Returns
    -------
    list
        List of all the 3D Sobel kernels (Sx, Sy, Sz, diagonal kernels).
    '''
    Sx = asarray(
        [[[-n1, 0, n1],
          [-n2, 0, n2],
          [-n1, 0, n1]],
         [[-n2, 0, n2],
          [-n3*n2, 0, n3*n2],
          [-n2, 0, n2]],
         [[-n1, 0, n1],
          [-n2, 0, n2],
          [-n1, 0, n1]]])

    Sy = asarray(
        [[[-n1, -n2, -n1],
          [0, 0, 0],
          [n1, n2, n1]],
         [[-n2, -n3*n2, -n2],
          [0, 0, 0],
          [n2, n3*n2, n2]],
         [[-n1, -n2, -n1],
          [0, 0, 0],
          [n1, n2, n1]]])

    Sz = asarray(
        [[[-n1, -n2, -n1],
          [-n2, -n3*n2, -n2],
          [-n1, -n2, -n1]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
         [[n1, n2, n1],
          [n2, n3*n2, n2],
          [n1, n2, n1]]])

    Sd11 = asarray(
        [[[0, n1, n2],
          [-n1, 0, n1],
          [-n2, -n1, 0]],
         [[0, n2, n2*n3],
          [-n2, 0, n2],
          [-n2*n3, -n2, 0]],
         [[0, n1, n2],
          [-n1, 0, n1],
          [-n2, -n1, 0]]])

    Sd12 = asarray(
        [[[-n2, -n1, 0],
          [-n1, 0, n1],
          [0, n1, n2]],
         [[-n2*n3, -n2, 0],
          [-n2, 0, n2],
          [0, n2, n2*n3]],
         [[-n2, -n1, 0],
          [-n1, 0, n1],
          [0, n1, n2]]])

    Sd21 = Sd11.T
    Sd22 = Sd12.T
    Sd31 = asarray([-S.T for S in Sd11.T])
    Sd32 = asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]

class GradEdge3D():
    '''
    Implements Sobel edge detection for 3D images using PyTorch.

    Parameters
    ----------
    n1 : int, optional
        Filter size for the first dimension (default is 1).
    n2 : int, optional
        Filter size for the second dimension (default is 2).
    n3 : int, optional
        Filter size for the third dimension (default is 2).
    '''

    def __init__(self, n1=1, n2=2, n3=2):
        super(GradEdge3D, self).__init__()
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        # Initialize Sobel filters for edge detection
        for s in S:
            sobel_filter = Conv3d(
                in_channels=1, out_channels=1, stride=1,
                kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = from_numpy(
                s.astype(float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(dtype=tfloat32)
            self.sobel_filters.append(sobel_filter)

    def __call__(self, img, a=1):
        '''
        Perform edge detection on the given 3D image.

        Parameters
        ----------
        img : torch.Tensor
            3D input tensor of shape (B, C, x, y, z).
        a : int, optional
            Padding size (default is 1).

        Returns
        -------
        torch.Tensor
            Tensor containing the gradient magnitudes of the edges.
        '''
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = tpad(img, pad, mode='reflect')

        # Calculate gradient magnitude of edges
        grad_mag = (1 / C) * tsum(stack([tsum(cat(
            [s.to(img.device)(img[:, c:c+1]) for c in range(C)],
            dim=1) + 1e-6, dim=1) ** 2 for s in self.sobel_filters],
            dim=1) + 1e-6, dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]

        num = grad_mag - grad_mag.min()
        dnm = grad_mag.max() - grad_mag.min() + 1e-6
        norm_grad_mag = num / dnm

        return norm_grad_mag.view(B, 1, H, W, D)

class GMELoss3D(Module):
    '''
    Implements Gradient Magnitude Edge Loss for 3D image data.

    Parameters
    ----------
    n1 : int
        Filter size for the first dimension.
    n2 : int
        Filter size for the second dimension.
    n3 : int
        Filter size for the third dimension.
    lam_errors : list
        List of tuples (weight, loss function) for computing error.
    reduction : str
        Reduction method for loss ('sum' or 'mean').
    '''

    def __init__(self, n1=1, n2=2, n3=2,
                 lam_errors=[(1.0, L1Loss())], reduction='sum'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3)
        self.lam_errors = lam_errors
        self.reduction = reduction

    def forward(self, x, y):
        '''
        Compute the loss based on the edges detected in the input tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, x, y, z).
        y : torch.Tensor
            Target tensor of shape (B, C, x, y, z).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        '''
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'

        edge_x = self.edge_filter(x)
        edge_y = self.edge_filter(y)

        if self.reduction == 'sum':
            error = 1e-6 + sum([lam * err_func(edge_x, edge_y) for lam, err_func in self.lam_errors])
        else:
            error = 1e-6 + (
                sum([lam * err_func(edge_x, edge_y) for lam, err_func in self.lam_errors]) / len(self.lam_errors))

        return error
    
# SSIM Loss (Structural Similarity)
def ssim_loss(pred, target, window_size=11, size_average=True, mask=None):
    """
    Computes SSIM loss between pred and target, optionally using a binary mask to ignore background.
    Supports 2D (4D tensor) and 3D (5D tensor) inputs.
    """
    if pred.dim() == 4:
        (_, channel, height, width) = pred.size()
        conv = F.conv2d
    elif pred.dim() == 5:
        (_, channel, depth, height, width) = pred.size()
        conv = F.conv3d
    else:
        raise ValueError("Expected input images to be 4D or 5D tensors")

    def create_window(window_size, channel):
        gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-gauss ** 2 / (2 * 1.5 ** 2))
        gauss = gauss / gauss.sum()

        if pred.dim() == 4:
            window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
            window = window_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
        else:
            window_3d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
            window_3d = window_3d.unsqueeze(2) * gauss.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            window = window_3d.unsqueeze(0).repeat(channel, 1, 1, 1, 1)

        return window

    window = create_window(window_size, channel).to(pred.device)

    mu_pred = conv(pred, window, padding=window_size // 2, groups=channel)
    mu_target = conv(target, window, padding=window_size // 2, groups=channel)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = conv(pred * pred, window, padding=window_size // 2, groups=channel) - mu_pred_sq
    sigma_target_sq = conv(target * target, window, padding=window_size // 2, groups=channel) - mu_target_sq
    sigma_pred_target = conv(pred * target, window, padding=window_size // 2, groups=channel) - mu_pred_target

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    if mask is not None:
        mask = mask.to(pred.device)

        if mask.shape != ssim_map.shape:
            mask = F.interpolate(mask.float(), size=ssim_map.shape[2:], mode='nearest')

        mask = mask.bool()

        if size_average:
            masked_ssim = ssim_map * mask
            mean_ssim = masked_ssim.sum() / mask.sum()
            return 1 - mean_ssim
        else:
            return 1 - ssim_map * mask
    else:
        return 1 - ssim_map.mean() if size_average else 1 - ssim_map

# Hybrid Loss Function
class HybridLossDynamic(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.7):
        """
        alpha: peso MSELoss
        beta:  peso SSIMLoss
        gamma: peso GMELoss
        """
        super(HybridLossDynamic, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gme = GMELoss3D()  

    def masked_voxelwise_mse(self, pred, target, mask):
        valid_pred = pred[mask.bool()]
        valid_target = target[mask.bool()]
        return F.mse_loss(valid_pred, valid_target)

    def forward(self, pred, target, mask=None):
        if mask is not None:
            loss_mse = self.masked_voxelwise_mse(pred, target, mask)
            loss_ssim = ssim_loss(pred, target, mask=mask)
        else:
            loss_mse = F.mse_loss(pred, target)
            loss_ssim = ssim_loss(pred, target)

        loss_gme = self.gme(pred, target)  

        total_loss = self.alpha * loss_mse + self.beta * loss_ssim + self.gamma * loss_gme
        return total_loss, loss_mse, loss_ssim, loss_gme


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch+1} to {path}")

# Custom function for loading NIfTI images
class NiftiDataset(Dataset):
    def __init__(self, target_img_paths, input_img_paths, mask_img_paths=None, target_shape=(64, 64, 64), cache=False):
        self.target_img_paths = target_img_paths
        self.input_img_paths = input_img_paths
        self.mask_img_paths = mask_img_paths
        self.target_shape = target_shape
        self.cache = cache

        if self.cache:  # Caching the data to reduce training time
            print("Caching images into memory...")
            t1 = time.time()
            self.cached_targets = []
            self.cached_inputs = []
            self.cached_masks = []

            for i, (t_path, i_paths) in enumerate(zip(self.target_img_paths, self.input_img_paths)):
                # === Target ===
                t_img = nib.load(t_path).get_fdata(dtype=np.float32)
                t_img = skimage.transform.resize(t_img, self.target_shape, mode='constant')
                self.cached_targets.append(torch.tensor(t_img, dtype=torch.float32).unsqueeze(0))  # [1, D, H, W]

                # === Inputs ===
                imgs = [nib.load(p).get_fdata(dtype=np.float32) for p in i_paths]
                imgs = [skimage.transform.resize(img, self.target_shape, mode='constant') for img in imgs]
                img_stack = np.stack(imgs, axis=0)  # [C, D, H, W]
                self.cached_inputs.append(torch.tensor(img_stack, dtype=torch.float32))

                # === Mask (optional) ===
                if self.mask_img_paths is not None:
                    mask_img = nib.load(self.mask_img_paths[i]).get_fdata(dtype=np.float32)
                    mask_img = skimage.transform.resize(mask_img, self.target_shape, mode='constant')
                    mask_tensor = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
                    self.cached_masks.append(mask_tensor)

            t2 = time.time()
            print(f"Done caching. Time: {t2 - t1:.2f} s")

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        if self.cache:
            input_tensor = self.cached_inputs[idx]
            target_tensor = self.cached_targets[idx]
            mask_tensor = self.cached_masks[idx] if self.mask_img_paths is not None else None
        else:
            # === Target ===
            target_img = nib.load(self.target_img_paths[idx]).get_fdata(dtype=np.float32)
            target_img = skimage.transform.resize(target_img, self.target_shape, mode='constant')
            target_tensor = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)

            # === Inputs ===
            input_imgs = [nib.load(path).get_fdata(dtype=np.float32) for path in self.input_img_paths[idx]]
            input_imgs = [skimage.transform.resize(img, self.target_shape, mode='constant') for img in input_imgs]
            input_tensor = torch.tensor(np.stack(input_imgs, axis=0), dtype=torch.float32)

            # === Mask ===
            if self.mask_img_paths is not None:
                mask_img = nib.load(self.mask_img_paths[idx]).get_fdata(dtype=np.float32)
                mask_img = skimage.transform.resize(mask_img, self.target_shape, mode='constant')
                mask_tensor = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0)
            else:
                mask_tensor = None

        return input_tensor, target_tensor, mask_tensor, self.input_img_paths[idx][0]  

in_chs = 1 #choose the number of parameters that you want to use

# Define a 3D U-Net architecture for volumetric data
class UNet3D(nn.Module):
    """
        Initialize the UNet3D model.

        Args:
            in_channels (int): Number of input channels (e.g., number of modalities).
            out_channels (int): Number of output channels 
            features (list): List of feature sizes for each encoder/decoder level.
            dropout_rate (float): Dropout rate used in the bottleneck for regularization.
    """
    def __init__(self, in_channels=in_chs, out_channels=1, features=[16, 32, 64], dropout_rate=0.3):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.dropout_rate = dropout_rate

        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))

        self.bottleneck = nn.Sequential(
            self.double_conv(features[-1], features[-1] * 2),
            nn.Dropout3d(p=self.dropout_rate)
        )

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=3, padding=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Define the forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape [B, C, D, H, W].

        Returns:
            Tensor: Output tensor of shape [B, out_channels, D, H, W].
        """
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skips[-(i // 2 + 1)]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i + 1](x)

        return self.final_conv(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SubjectsFolder = '/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/InputChannels'  # Path to MUDI parameters
OutFolder = '/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/ablation_nopreweight_masked3.0' # Choose accordingly to your path
MRE_Folder = '/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/MRE_T1toMNI_202402'            # Choose accordingly to your path

ListofSubjectNames = [
    os.path.join(SubjectsFolder, d)
    for d in os.listdir(SubjectsFolder)
    if os.path.isdir(os.path.join(SubjectsFolder, d)) 
       and ('PD20' in d or 'PD_20' in d) 
       and 'aug' not in d
]

print("Number of subj:", len(ListofSubjectNames))

all_parameters = ['dtd_covariance_C_mu_to_t1_antstrans_t1_to_MNI_median_normalizedxparam.nii.gz']

'''
    'dtd_covariance_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_covariance_C_mu_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_covariance_FA_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_covariance_V_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_codivide_md_t_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_codivide_v_at_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_codivide_v_fw_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    't2_to_t1_t1_to_MNI_normalizedxparam.nii.gz',
    'T1_t1_to_MNI_normalizedxparam.nii.gz'
]
'''
# Hyperparameters
batch_size = 3
lr = 0.0001 
num_epochs = 4000 
patience=100
es_warmup=50
#min_delta = 1e-03, # 1e-03

target_shape = (64, 64, 64)  # Adjust based on memory constraints and model requirements
all_results = []


def save_prediction_as_nifti(prediction, reference_nifti_path, output_filename):
    """
    Save a resized PyTorch tensor prediction as a NIfTI file with the affine transformation
    from a reference NIfTI file.

    Args:
        prediction (torch.Tensor): The model's predicted output tensor.
        reference_nifti_path (str): Path to the original NIfTI file for resizing and affine.
        output_filename (str): Path to save the output NIfTI file.
    """
    # Convert prediction to numpy and remove any singleton dimensions
    prediction_data = prediction.squeeze().cpu()

    # Load the reference NIfTI to get the target shape and affine matrix
    reference_img = nib.load(str(reference_nifti_path))
    reference_shape = reference_img.shape
    affine = reference_img.affine
    header = reference_img.header

    # Resize the prediction to match the reference image shape
    prediction_resized = F.interpolate(prediction_data.unsqueeze(0).unsqueeze(0), 
                                       size=reference_shape, mode='trilinear', align_corners=False)
    prediction_resized = prediction_resized.squeeze().numpy()  # Remove extra dimensions for saving

    # Create a NIfTI image with the resized prediction data and reference affine
    prediction_nifti = nib.Nifti1Image(prediction_resized, affine, header)
    
    # Save the new NIfTI image
    nib.save(prediction_nifti, output_filename)
    print(f"Saved prediction to {output_filename}")
    
'''
    'dtd_covariance_C_mu_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_covariance_FA_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_covariance_V_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_codivide_md_t_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_codivide_v_at_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    'dtd_codivide_v_fw_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
    't2_to_t1_t1_to_MNI_normalizedxparam.nii.gz',
    'T1_t1_to_MNI_normalizedxparam.nii.gz'
]
'''
mask_paths = []
input_img_paths = []
target_img_paths = []
mask_paths_PD = []
input_img_paths_PD = []
target_img_paths_PD = []

for SubjectName in ListofSubjectNames:
    SubjectName = os.path.basename(SubjectName)
    SubjectSpaceFolderOut = os.path.join(SubjectsFolder, SubjectName)

    relative_pathnames = ['dtd_covariance_C_mu_to_t1_antstrans_t1_to_MNI_median_normalizedxparam.nii.gz']
    full_pathnames = [os.path.join(SubjectSpaceFolderOut, rel_path) for rel_path in relative_pathnames]
    all_exist = all(os.path.exists(path) for path in full_pathnames)

    mrepath = os.path.join(MRE_Folder, SubjectName, 'MRE_stiffness_ToT1_202402_t1_to_MNI_normalized.nii.gz')
    mask_path = os.path.join(MRE_Folder, SubjectName, 'MRE_stiffness_mask.nii.gz') 

    if os.path.exists(mrepath) and os.path.exists(mask_path) and all_exist:
        if 'Control' in SubjectName:
            input_img_paths.append(full_pathnames)
            target_img_paths.append(mrepath)
            mask_paths.append(mask_path)
        elif 'PD' in SubjectName:
            input_img_paths_PD.append(full_pathnames)
            target_img_paths_PD.append(mrepath)
            mask_paths_PD.append(mask_path)
    else:
        print(f"Skipping {SubjectName}: missing files.")

in_chs = len(relative_pathnames)
print(in_chs)

# Create dataset and dataloader
# === Split: 13 HC + 8 PD for training, 2 HC + 2 PD for validation, 2 HC + 2 PD for testing ===
# === Train ===
train_targets = target_img_paths[:13] +  target_img_paths_PD[:8]
train_inputs  = input_img_paths[:13] + input_img_paths_PD[:8] 
train_masks   = mask_paths[:13] + mask_paths_PD[:8]

# === Validation ===
val_targets = target_img_paths[13:15] + target_img_paths_PD[8:10]
val_inputs  = input_img_paths[13:15] + input_img_paths_PD[8:10]
val_masks   = mask_paths[13:15] + mask_paths_PD[8:10]

# === Test ===
test_targets = target_img_paths[15:17] + target_img_paths_PD[10:12]
test_inputs  = input_img_paths[15:17] + input_img_paths_PD[10:12]
test_masks   = mask_paths[15:17] + mask_paths_PD[10:12]

# === Create datasets with masks ===
train_dataset = NiftiDataset(train_targets, train_inputs, mask_img_paths=train_masks, target_shape=target_shape, cache=True)
val_dataset   = NiftiDataset(val_targets, val_inputs, mask_img_paths=val_masks, target_shape=target_shape, cache=True)
test_dataset  = NiftiDataset(test_targets, test_inputs, mask_img_paths=test_masks, target_shape=target_shape, cache=True)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

# === DataLoader ===
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

experiment_name = f'C_mu'

writer = SummaryWriter(log_dir=os.path.join(OutFolder, experiment_name, "runs"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=UNet3D(in_chs, 1, [32, 64, 128]).to(device)

criterion=HybridLossDynamic(alpha=1.0, beta=1.0, gamma=1.0)
weight_decay = 1e-4
optimizer = optim.Adam( model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay  # Ensure this is defined
                    )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
# Load checkpoint if resuming training
experiment_out_dir = os.path.join(OutFolder, experiment_name)
bestmodel_path = os.path.join(experiment_out_dir, "bestmodel_checkpoint.pth") 

'''
#To resume from the checkpoint
#resume_from_dir = "/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/ProveModel/Edge9params_reshape2/9params_training_EarlyStop" 
resume_best_path = os.path.join(resume_from_dir, "bestmodel_checkpoint.pth")

if os.path.exists(resume_best_path):
    start_epoch, best_val_loss, best_model_state = load_checkpoint_partial(model, path=resume_best_path)
    no_improve_epochs = 0
    print("Loaded best model weights for fine-tuning / ablation.")
else:
    print("Best model checkpoint not found, training from scratch.")
    start_epoch = 0
'''    

start_epoch = 0
best_val_loss = float('inf')
best_model_state = None
no_improve_epochs = 0

#  --- Training loop ---
for epoch in range(start_epoch, num_epochs):  
    model.train()
    note = ""
    train_loss = 0
    train_ssim = 0
    train_mse = 0
    train_gme = 0
    
    print(f'\nEpoch {epoch + 1}/{num_epochs}')

    # Progress bar for the epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, ncols=110)  
    epoch_loss = []  

    for inputs, targets, masks, filenames in pbar:
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        outputs = model(inputs)
    
        loss_train, mse_loss_train, ssim_loss_train, gme_loss_train = criterion(outputs, targets, mask=masks) 

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        train_loss += loss_train.item()
        train_ssim += ssim_loss_train.item()  
        train_mse += mse_loss_train.item() 
        train_gme += gme_loss_train.item()

        pbar.set_postfix({'hybrid_loss': loss_train.item(),
                          'mse': mse_loss_train.item(),
                          'ssim': ssim_loss_train.item(),
                          'gme': gme_loss_train.item()})
    
    train_loss /= len(train_loader) 
    train_mse /= len(train_loader)
    train_ssim /= len(train_loader)  
    train_gme /= len(train_loader) 

    writer.add_scalar("Loss/Train", train_loss, epoch) 
    writer.add_scalar("Metrics/MSE/Train", train_mse, epoch)
    writer.add_scalar("Metrics/SSIM/Train", train_ssim, epoch) 
    writer.add_scalar("Metrics/GME/Train", train_gme, epoch)  
    writer.add_scalar("Metrics/LR", lr, epoch)

    model.eval()
    val_loss = 0
    val_ssim = 0
    val_mse = 0
    val_gme = 0

    with torch.no_grad():
        for inputs, targets, masks, filenames in val_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            outputs = model(inputs)

            # Calculation of loss
            loss_val, mse_loss_val, ssim_loss_val, gme_loss_val = criterion(outputs, targets, mask=masks)
            val_loss += loss_val.item()
            val_mse += mse_loss_val.item()
            val_ssim += ssim_loss_val.item()
            val_gme += gme_loss_val.item()


    # Calculation of averages
    val_loss /= len(val_loader)
    val_mse /= len(val_loader)
    val_ssim /= len(val_loader)
    val_gme /= len(val_loader)

    # Logging to TensorBoard
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Metrics/MSE/Validation", val_mse, epoch)
    writer.add_scalar("Metrics/SSIM/Validation", val_ssim, epoch)
    writer.add_scalar("Metrics/GME/Validation", val_gme, epoch)
    
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]['lr']

    # Print MSE and SSIM values at the end of the epoch
    print(f"Training  -  Epoch {epoch + 1}: loss_MSE: {train_mse:.4f},loss_SSIM: {train_ssim:.4f},loss_GME: {train_gme:.4f}, Total Loss: {train_loss:.4f} - lr: {lr:.6f}")
    # Print to console
    print(f"Validation - loss_MSE: {val_mse:.4f},loss_SSIM: {val_ssim:.4f},loss_GME: {val_gme:.4f}, Total Loss: {val_loss:.4f}")

    # --- Early Stopping ---
    if es_warmup < epoch:
        if val_loss < best_val_loss:
            note += "ES"
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
            best_epoch = epoch
            torch.save({"model_state_dict": best_model_state,"best_val_loss": best_val_loss,"no_improve_epochs": no_improve_epochs}, bestmodel_path)
        else:
            no_improve_epochs += 1
            torch.save({"model_state_dict": best_model_state,"best_val_loss": best_val_loss,"no_improve_epochs": no_improve_epochs}, bestmodel_path)
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Directory to save NIfTI predictions
nifti_output_dir = os.path.join(experiment_out_dir, "niftimaps")
os.makedirs(nifti_output_dir, exist_ok=True)

model.eval()
mse_list = []
ssim_list = []
psnr_list = []

global_max_val = 1

def calculate_psnr(mse, max_val=global_max_val):
    if mse == 0:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)

with torch.no_grad():
    for inputs, targets, masks, filenames in test_loader:
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        outputs = model(inputs)

        # if necessary reshape the mask
        if masks.shape != outputs.shape:
            resized_mask = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
        else:
            resized_mask = masks

        valid_outputs = outputs[resized_mask.bool()]
        valid_targets = targets[resized_mask.bool()]

        mse_value = F.mse_loss(valid_outputs, valid_targets).item()
        ssim_value = 1 - ssim_loss(outputs, targets, mask=resized_mask).item()
        psnr_value = calculate_psnr(mse_value, max_val=global_max_val)

        mse_list.append(mse_value)
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

        # Save NIfTI
        for i, output in enumerate(outputs):
            reference_path = filenames[i]
            subject_filename = Path(reference_path).parent.name
            output_filename = os.path.join(nifti_output_dir, f"{subject_filename}_prediction.nii.gz")
            save_prediction_as_nifti(output, reference_path, output_filename)


# Mean and STD
mean_mse = np.mean(mse_list)
std_mse  = np.std(mse_list)

mean_ssim = np.mean(ssim_list)
std_ssim  = np.std(ssim_list)

mean_psnr = np.mean(psnr_list)
std_psnr  = np.std(psnr_list)

print(f"Test - MSE: {mean_mse:.4f} ± {std_mse:.4f}, SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}, PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")

# CSV output
evaluation_results = [{
    "experiment": experiment_name,
    "epochsaved": best_epoch,
    "test_mse_mean": mean_mse,
    "test_mse_std": std_mse,
    "test_ssim_mean": mean_ssim,
    "test_ssim_std": std_ssim,
    "test_psnr_mean": mean_psnr,
    "test_psnr_std": std_psnr
}]
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv(os.path.join(experiment_out_dir, "evaluation_results.csv"), index=False)