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

from numpy import asarray, float32
from torch import from_numpy, sum as tsum, stack, cat, float32 as tfloat32
from torch.nn import Module, Conv3d, L1Loss
from torch.nn.functional import pad as tpad
import math

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
def ssim_loss(pred, target, window_size=11, size_average=True):
    """
    Computes the SSIM loss between `pred` and `target` for both 2D and 3D images.
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

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1 - ssim_map.mean() if size_average else 1 - ssim_map

# Hybrid Loss Function
class HybridLossDynamic(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.7):
        '''
         A hybrid loss that combines:
        - Mean Squared Error (MSE)
        - Structural Similarity Index (SSIM)
        - Gradient Magnitude Edge (GME) loss

        The weights for each component are tunable via alpha, beta, and gamma.
        '''
        super(HybridLossDynamic, self).__init__()
        self.mse = nn.MSELoss()
        self.gme = GMELoss3D()  # già definita nel tuo codice
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
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


'''
# two functions to load checkpoint in case you want to use pre-trained model

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
    return start_epoch

def load_checkpoint_partial(model, path="checkpoint.pth"):
    """
    Loads only the compatible layers of a model from a checkpoint.
    Useful for fine-tuning or when the architecture has changed slightly.

    Parameters:
    - model (torch.nn.Module): The model to load weights into.
    - path (str): Path to the checkpoint file.

    Returns:
    - start_epoch (int): Set to 0 (since training is restarted).
    - best_val_loss (float): The best validation loss saved in the checkpoint.
    - best_model_state (dict): The current updated state dict of the model.
    """
    checkpoint = torch.load(path, map_location="cpu")
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()

    # Filter out only the compatible weights (same keys and shapes)
    compatible_weights = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    # Keep track of skipped layers for reporting
    skipped = [
        k for k in pretrained_dict
        if k not in compatible_weights
    ]

    print(f" Loaded {len(compatible_weights)} compatible layer.")
    if skipped:
        print(f" skipped {len(skipped)} layer for incompatibility:")

    # Update model state with compatible weights
    model_dict.update(compatible_weights)
    model.load_state_dict(model_dict)

    # Do not load optimizer state; training will restart from scratch
    start_epoch = 0
    best_val_loss = checkpoint['best_val_loss']
    best_model_state = model.state_dict()
    no_improve_epochs = 0  # Not returned, but might be used later

    return start_epoch, best_val_loss, best_model_state
'''

# Custom dataset for loading NIfTI images
class NiftiDataset(Dataset):
    def __init__(self, target_img_paths, input_img_paths, target_shape=(64, 64, 64), cache=False):
        """
        Initializes the dataset.

        Parameters:
        - target_img_paths (list): List of file paths for target images (e.g., ground truth).
        - input_img_paths (list of lists): List of lists of input image paths (e.g., modalities).
        - target_shape (tuple): Desired shape to resize all volumes to (default is 64x64x64).
        - cache (bool): If True, loads all data into memory at initialization to speed up training.
        """
        self.target_img_paths = target_img_paths
        self.input_img_paths = input_img_paths
        self.target_shape = target_shape
        self.cache = cache

        if self.cache:  # Optional caching to avoid loading files repeatedly
            print("Caching images into memory...")
            t1 = time.time()
            self.cached_targets = []
            self.cached_inputs = []

            # Iterate over all image paths
            for t_path, i_paths in zip(self.target_img_paths, self.input_img_paths):
                # Load and resize target image
                t_img = nib.load(t_path).get_fdata(dtype=np.float32)
                t_img = skimage.transform.resize(t_img, self.target_shape, mode='constant')
                self.cached_targets.append(torch.tensor(t_img, dtype=torch.float32).unsqueeze(0))  # Add channel dimension

                # Load and resize input images (e.g., multiple modalities)
                imgs = [nib.load(p).get_fdata(dtype=np.float32) for p in i_paths]
                imgs = [skimage.transform.resize(img, self.target_shape, mode='constant') for img in imgs]
                img_stack = np.stack(imgs, axis=0)  # Stack input channels
                self.cached_inputs.append(torch.tensor(img_stack, dtype=torch.float32))
            t2 = time.time()
            print(t2 - t1)
            print("Done caching.")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        """
        Loads and returns the input tensor, target tensor, and file name for a given index.

        If caching is enabled, data is retrieved from memory; otherwise, it's loaded from disk.

        Returns:
        - input_tensor (Tensor): The stacked input modalities (C, H, W, D).
        - target_tensor (Tensor): The ground truth image (1, H, W, D).
        - filename (str): The filename of the first input image for reference.
        """
        if self.cache:
            input_tensor = self.cached_inputs[idx]
            target_tensor = self.cached_targets[idx]
        else:
            # Load and process target image
            target_img = nib.load(self.target_img_paths[idx]).get_fdata(dtype=np.float32)
            target_img = skimage.transform.resize(target_img, self.target_shape, mode='constant')
            target_tensor = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)

            # Load and process input images
            input_imgs = [nib.load(path).get_fdata(dtype=np.float32) for path in self.input_img_paths[idx]]
            input_imgs = [skimage.transform.resize(img, self.target_shape, mode='constant') for img in input_imgs]
            input_tensor = torch.tensor(np.stack(input_imgs, axis=0), dtype=torch.float32)

        return input_tensor, target_tensor, self.input_img_paths[idx][0]

in_chs = 6 #choose the number of parameters that you want to use


# Define a 3D U-Net architecture for volumetric data
class UNet3D(nn.Module):
    def __init__(self, in_channels=in_chs, out_channels=1, features=[16, 32, 64], dropout_rate=0.3):
        """
        Initialize the UNet3D model.

        Args:
            in_channels (int): Number of input channels (e.g., number of modalities).
            out_channels (int): Number of output channels 
            features (list): List of feature sizes for each encoder/decoder level.
            dropout_rate (float): Dropout rate used in the bottleneck for regularization.
        """
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
        """
        A block of two 3D convolutions with ReLU activations.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        Returns:
            nn.Sequential: Two Conv3D + ReLU layers.
        """
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
OutFolder = '/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/ablation_nopreweight' # Choose accordingly to your path
MRE_Folder = '/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/MRE_T1toMNI_202402' # Choose accordingly to your path

ListofSubjectNames = [
    os.path.join(SubjectsFolder, d)
    for d in os.listdir(SubjectsFolder)
    if os.path.isdir(os.path.join(SubjectsFolder, d)) 
       and ('PD20' in d or 'PD_20' in d) 
       and 'aug' not in d
]
print("Number of subj:", len(ListofSubjectNames))

all_parameters = ['dtd_covariance_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_covariance_FA_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_covariance_V_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_codivide_md_t_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_codivide_v_at_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_codivide_v_fw_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz']

'''
# All different parameters available:
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

target_shape = (64, 64, 64)  # Adjust based on memory constraints and model requirements
#target_shape = (128,128, 128)  # Adjust based on memory constraints and model requirements
#target_shape = (160, 192, 160)
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
    

input_img_paths = []
target_img_paths = []
input_img_paths_PD = []
target_img_paths_PD = []

# Iterate across all the patients
for SubjectName in ListofSubjectNames:
    SubjectName = os.path.basename(SubjectName)
    SubjectSpaceFolderOut = os.path.join(SubjectsFolder, SubjectName)

    relative_pathnames = ['dtd_covariance_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_covariance_FA_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_covariance_V_MD_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_codivide_md_t_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_codivide_v_at_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz',
                          'dtd_codivide_v_fw_to_t1_antstrans_t1_to_MNI_normalizedxparam.nii.gz']

    full_pathnames = [os.path.join(SubjectSpaceFolderOut, rel_path) for rel_path in relative_pathnames]
    all_exist = all(os.path.exists(path) for path in full_pathnames)

    mrepath = os.path.join(MRE_Folder, SubjectName, 'MRE_stiffness_ToT1_202402_t1_to_MNI_normalized.nii.gz')

    if os.path.exists(mrepath) and all_exist:
        if 'Control' in SubjectName:
            input_img_paths.append(full_pathnames)
            target_img_paths.append(mrepath)
        elif 'PD' in SubjectName:
            input_img_paths_PD.append(full_pathnames)
            target_img_paths_PD.append(mrepath)
    else:
        print(f"Skipping {SubjectName}: missing files.")

in_chs = len(relative_pathnames)
print(in_chs)

# Create dataset and dataloader
# === Split: 13 HC + 8 PD for training, 2 HC + 2 PD for validation, 2 HC + 2 PD for testing ===
# Train
train_targets = target_img_paths[:13] + target_img_paths_PD[:8]
train_inputs  = input_img_paths[:13] + input_img_paths_PD[:8]

# Validation
val_targets = target_img_paths[13:15] + target_img_paths_PD[8:10]
val_inputs  = input_img_paths[13:15] + input_img_paths_PD[8:10]

# Test
test_targets = target_img_paths[15:17] + target_img_paths_PD[10:12]
test_inputs  = input_img_paths[15:17] + input_img_paths_PD[10:12]

# === Create Dataset and DataLoader ===
train_dataset = NiftiDataset(train_targets, train_inputs, target_shape=target_shape, cache=True)
val_dataset   = NiftiDataset(val_targets, val_inputs, target_shape=target_shape, cache=True)
test_dataset  = NiftiDataset(test_targets, test_inputs, target_shape=target_shape, cache=True)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


experiment_name = f'JustDiffusion_WOCmu'

writer = SummaryWriter(log_dir=os.path.join(OutFolder, experiment_name, "runs"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=UNet3D(in_chs, 1, [32, 64, 128]).to(device)

criterion=HybridLossDynamic(alpha=1.0, beta=1.0, gamma=1.0)
weight_decay = 1e-4
optimizer = optim.Adam( model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay  # Ensure this is defined
                    )

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

experiment_out_dir = os.path.join(OutFolder, experiment_name)
bestmodel_path = os.path.join(experiment_out_dir, "bestmodel_checkpoint.pth")
 
# To resume from the checkpoint
'''
resume_from_dir = "/home/maia-user/cifs/Datasets/PD_Private/chrol/ParkMRE/ProveModel/Edge9params_reshape2/9params_training_EarlyStop" 
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

#  --- TRAINING loop ---
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

    for inputs, targets, filenames in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
    
        loss_train, mse_loss_train, ssim_loss_train, gme_loss_train = criterion(outputs, targets) 

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
        for inputs, targets, filenames in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Calculation of loss
            loss_val, mse_loss_val, ssim_loss_val, gme_loss_val = criterion(outputs, targets)
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

# --- TEST PHASE ---
model.eval()

# Initialize accumulators for metrics
test_ssim = 0
test_mse = 0
test_psnr = 0
global_max_val = 0.0

# Updated PSNR function
def calculate_psnr(mse, max_val=1):
    if mse == 0:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)

with torch.no_grad():
    for inputs, targets, filenames in test_loader:
        # Move input and target to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute evaluation metrics
        mse_value = F.mse_loss(outputs, targets).item()
        ssim_value = 1 - ssim_loss(outputs, targets).item()  # Convert SSIM loss to SSIM score
        psnr_value = calculate_psnr(mse_value, max_val=global_max_val)

        test_mse += mse_value
        test_ssim += ssim_value
        test_psnr += psnr_value

        # Save predictions as NIfTI files
        for i, output in enumerate(outputs):
            reference_path = filenames[i]
            subject_filename = Path(reference_path).parent.name
            output_filename = os.path.join(nifti_output_dir, f"{subject_filename}_prediction_{i}.nii.gz")
            save_prediction_as_nifti(output, reference_path, output_filename)

# Compute average metrics across the test dataset
test_mse /= len(test_loader)
test_ssim /= len(test_loader)
test_psnr /= len(test_loader)

# Print evaluation results to the console
print(f"Test - SSIM: {test_ssim:.4f}, MSE: {test_mse:.4f}, PSNR: {test_psnr:.2f} dB")

# Save evaluation results to CSV format
evaluation_results = [{
    "experiment": experiment_name,
    "epochsaved": best_epoch,
    "test_ssim": test_ssim,
    "test_mse": test_mse,
    "test_psnr": test_psnr
}]

eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv(os.path.join(experiment_out_dir, "evaluation_results.csv"), index=False)