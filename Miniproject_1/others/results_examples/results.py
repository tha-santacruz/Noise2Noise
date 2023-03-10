import re
import sys
import unittest
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

import pandas as pd
import matplotlib.pyplot as plt

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--project-path', help='Path to the project folder', required=True)
    parser.add_argument('-d', '--data-path', help='Path to the data folder', required=True)
    args = parser.parse_args()
    
    project_path = Path(args.project_path)
    data_path = Path(args.data_path)

    if re.match(r'^Proj(_(\d{6})){3}$', project_path.name) is None:
        warn("Project folder name must be in the form Proj_XXXXXX_XXXXXX_XXXXXX")

    sys.path.append(args.project_path)

    # data demo
    Model = importlib.import_module("Miniproject_1.model").Model
    model = Model()
    model.load_pretrained_model(model_name= "bestmodel.pth")
    val_path = data_path / "val_data.pkl"
    val_input, val_target = torch.load(val_path)

    input = val_input[:20].float()
    target = val_target[:20].float()
    with torch.no_grad():
        clean_output = model.predict(input).to(torch.int64)



    for i in range(20):
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(input[i].int().cpu().detach().permute(1,2,0).numpy())
        axarr[0].set_title("Noisy input")
        axarr[1].imshow(clean_output[i].cpu().detach().permute(1,2,0).numpy())
        axarr[1].set_title("Denoised output")
        axarr[2].imshow(target[i].int().cpu().detach().permute(1,2,0).numpy())
        axarr[2].set_title("Clean target")
        f.tight_layout()
        f.savefig(f"results_1/results{i}.png", bbox_inches = 'tight', pad_inches = 0)


    """i = 10
    f, axarr = plt.subplots(4,3)
    axarr[0,0].imshow(input[4*i].int().cpu().detach().permute(1,2,0).numpy())
    axarr[0,1].imshow(clean_output[4*i].cpu().detach().permute(1,2,0).numpy())
    axarr[0,2].imshow(target[4*i].cpu().detach().permute(1,2,0).numpy())
    axarr[1,0].imshow(input[4*i+1].cpu().detach().permute(1,2,0).numpy())
    axarr[1,1].imshow(clean_output[4*i+1].cpu().detach().permute(1,2,0).numpy())
    axarr[1,2].imshow(target[4*i+1].cpu().detach().permute(1,2,0).numpy())
    axarr[2,0].imshow(input[4*i+2].cpu().detach().permute(1,2,0).numpy())
    axarr[2,1].imshow(clean_output[4*i+2].cpu().detach().permute(1,2,0).numpy())
    axarr[2,2].imshow(target[4*i+2].cpu().detach().permute(1,2,0).numpy())
    axarr[3,0].imshow(input[4*i+3].cpu().detach().permute(1,2,0).numpy())
    axarr[3,1].imshow(clean_output[4*i+3].cpu().detach().permute(1,2,0).numpy())
    axarr[3,2].imshow(target[4*i+3].cpu().detach().permute(1,2,0).numpy())
    plt.show()"""
