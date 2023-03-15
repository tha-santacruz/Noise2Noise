import re
import sys
import importlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

import wandb

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

def train_model(lr, betas, eps, lossf, ndata, augment, project_number = 1):
    #run = wandb.init(project="noise2noise_10000_data", entity="bbk_2022", reinit=True)
    #run.name = f"lr={lr}, betas={betas}, eps={eps}, lossf={lossf.__name__}, ndata={ndata}"
    #run = wandb.init(project="noise2noise_GAN_50000", entity="bbk_2022", reinit=True)
    #run.name = f"lr={lr}, augment={augment}"
    #wandb.config = {
    #    "learning_rate": lr,
    #    "betas": betas,
    #    "epsilon": eps,
    #    "loss function": lossf,
    #    "num_data": ndata,
    #    "data_augmentation": augment
    #    }

    run = wandb.init(project="noise2noise", anonymous="must", reinit=True, config={
        "learning_rate": lr,
        "betas": betas,
        "epsilon": eps,
        "loss function": lossf,
        "num_data": ndata,
        "data_augmentation": augment,
        "batch_size": 10,
        })
    criterion = lossf()
    num_train_sessions = 100
    Model = importlib.import_module(f"Miniproject_{project_number}.model_tuning").Model
    model = Model(lr=lr,betas=betas,eps=eps,lossf=lossf, augment=augment)

    train_path = data_path / "train_data.pkl"
    val_path = data_path / "val_data.pkl"
    train_input0, train_input1 = torch.load(train_path)
    train_input0= train_input0[:ndata]
    train_input1= train_input1[:ndata]
    val_input, val_target = torch.load(val_path)
    val_target = val_target.float()

    mini_batch_size = 10

    for session in tqdm(range(num_train_sessions)):
        train_logs = model.train(train_input0, train_input1, num_epochs=1)

        model_outputs = []
        with torch.no_grad():
            for b in range(0, val_input.size(0), mini_batch_size) :
                output = model.predict(val_input.narrow(0, b, mini_batch_size))
                model_outputs.append(output.cpu())
            model_outputs = torch.cat(model_outputs, dim=0)
            val_loss = criterion(model_outputs, val_target)

        output_psnr_after = compute_psnr(model_outputs.div(255), val_target.div(255))
        train_logs = torch.cat((train_logs,torch.full((train_logs.size(0),1),float(output_psnr_after))),dim=1)
        wandb.log({
            "epoch": train_logs[-1,0],
            "time": train_logs[-1,2],
            "train_loss": train_logs[-1,1],
            "val_loss": val_loss,
            "val_PSNR": output_psnr_after})
        model.save_trained_model(model_name=f"checkpoints/checkpoint_number_{session}.pth")
        
    run.finish()

if __name__ == '__main__':
    project_path = Path("Project")
    data_path = Path("Miniproject_data")

    sys.path.append("Project")
    hyperparameters = {
        "lr":[0.00005],
        "betas": [(0.9, 0.99)],
        "eps": [1e-8],
        "lossf": [nn.MSELoss],
        "augment": [True]}
    num_data = 50000

    # First parameters set for recall
    for lr in hyperparameters["lr"]:
        for betas in hyperparameters["betas"]:
            for eps in hyperparameters["eps"]:
                for lossf in hyperparameters["lossf"]:
                    for augment in hyperparameters["augment"]:
                        print(f"training with lr {lr}, betas {betas}, eps {eps}, lossf {lossf.__name__} on {num_data} data examples")
                        train_model(lr=lr,betas=betas,eps=eps,lossf=lossf, ndata= num_data, augment=augment)