"""
DANFlow: Depth-wise separable convolution and Attention-based Normalizing Flow
Main training and evaluation script for anomaly detection on industrial datasets.
"""

import argparse
import os
import random
from copy import deepcopy
from glob import glob

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from ignite.contrib import metrics
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format
import dataset
import utils
from model import DANFlow
from model import constants as const

# Configuration
LOG_PATH = "log"
IMG_DIR = "img"


def set_seed(seed=25):
    """
    Set random seed for reproducibility across numpy, random, and PyTorch.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")


# Initialize random seed
set_seed(25)


def build_train_data_loader(args, config):
    dataset_name = getattr(dataset, args.dataset_name.upper() + "Dataset")
    train_dataset = dataset_name(
        root=args.data_path,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    print(f"Batch size: {const.BATCH_SIZE}")
    
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    dataset_name = getattr(dataset, args.dataset_name.upper() + "Dataset")
    test_dataset = dataset_name(
        root=args.data_path,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )


def build_model(config, attention):
    model = DANFlow(
        input_size=config["input_size"],
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        num_dw=config["num_dw"],
        double_subnets=config["double_subnets"],
        attention=attention
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model A.D. Param#: {num_params}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, config["input_size"],  config["input_size"])
    macs, params = profile(model, inputs=(dummy_input,))
    macs, params = clever_format([macs, params], "%.3f")
    
    print(f"MACs: {macs}")
    print(f"Total params: {params}")

    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=const.LR,
        weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch, writer):
    """
    Train model for one epoch.
    
    Args:
        dataloader: Training data loader
        model: Model to train
        optimizer: Optimizer
        epoch (int): Current epoch number
        writer: TensorBoard writer
    """
    model.train()
    loss_meter = utils.AverageMeter()
    n_total_steps = len(dataloader)

    for step, data in enumerate(dataloader):
        # Forward pass
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item())
        
        # Logging
        if step % const.LOG_INTERVAL == 0 or step == len(dataloader):
            print(f"Epoch {epoch} - Step {step}: loss = {loss_meter.val:.3f}({loss_meter.avg:.3f})")

            global_step = epoch * n_total_steps + step
            writer.add_scalar('training loss', loss_meter.val, global_step)
            writer.add_scalar('avg training loss', loss_meter.avg, global_step)

            # Log gradients
            grads_avg_dict = {}
            grads_std_dict = {}
            grads_max_dict = {}
            
            for name, params in model.named_parameters():
                if params.requires_grad and "bias" not in name:
                    grads_avg_dict[name] = params.grad.cpu().detach().mean()
                    grads_std_dict[name] = params.grad.cpu().detach().std()
                    grads_max_dict[name] = params.grad.cpu().detach().max()

            writer.add_scalars('average grads', grads_avg_dict, global_step)
            writer.add_scalars('std grads', grads_std_dict, global_step)
            writer.add_scalars('max grads', grads_max_dict, global_step)

            # Log weights
            weight_avg_dict = {}
            weight_std_dict = {}
            
            for name, params in model.state_dict().items():
                if 'weight' in name and 'nf_flows' in name:
                    weight_avg_dict[name] = params.cpu().detach().mean()
                    weight_std_dict[name] = params.cpu().detach().std()

            writer.add_scalars('average weight', weight_avg_dict, global_step)
            writer.add_scalars('std weight', weight_std_dict, global_step)


def eval_once(dataloader, model):
    """
    Evaluate model once and compute AUROC.
    
    Args:
        dataloader: Test data loader
        model: Model to evaluate
        
    Returns:
        float: AUROC score
    """
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        
        with torch.no_grad():
            ret = model(data)
            
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten().type(torch.IntTensor)
        
        auroc_metric.update((outputs, targets))
    
    auroc = auroc_metric.compute()
    print(f"AUROC: {auroc}")
    
    return auroc


def eval_test_images(dataloader, model, category):
    """
    Evaluate model on test images and generate visualizations.
    
    Args:
        dataloader: Test data loader
        model: Model to evaluate
        category (str): Category name
        
    Returns:
        float: AUROC score
    """
    # Create output directory
    os.makedirs(IMG_DIR, exist_ok=True)
    img_dir = os.path.join(IMG_DIR, f"exp{len(os.listdir(IMG_DIR))}_{category}")
    os.makedirs(img_dir, exist_ok=True)

    model.eval()
    auroc_metric = metrics.ROC_AUC()

    # Accumulate results
    final_datas = torch.Tensor().cuda()
    final_targets = torch.Tensor().cuda()
    final_outputs = torch.Tensor().cpu()
    final_outputs_flatten = torch.Tensor()
    final_targets_flatten = torch.IntTensor()

    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        
        with torch.no_grad():
            ret = model(data)

        outputs = ret["anomaly_map"].cpu().detach()
        outputs_flat = outputs.flatten()
        targets_flat = targets.flatten().type(torch.IntTensor)

        final_datas = torch.cat((final_datas, data), 0)
        final_targets = torch.cat((final_targets, targets), 0)
        final_outputs = torch.cat((final_outputs, outputs), 0)
        final_outputs_flatten = torch.cat((final_outputs_flatten, outputs_flat), 0)
        final_targets_flatten = torch.cat((final_targets_flatten, targets_flat), 0)

        auroc_metric.update((outputs_flat, targets_flat))

    auroc = auroc_metric.compute()
    print(f"AUROC: {auroc}")

    # Calculate optimal threshold using F1 score
    precision, recall, thresholds = precision_recall_curve(
        final_targets_flatten, final_outputs_flatten
    )
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # Generate visualizations
    count = 0
    for img, output, target in zip(final_datas, final_outputs, final_targets):
        original_im = utils.denormalize(img).permute(1, 2, 0).cpu().detach().numpy()
        pred = output[0].cpu().detach().numpy()
        target = target.type(torch.IntTensor)
        label = target.permute(1, 2, 0).cpu().detach().numpy()
        
        count += 1
        saved_path = os.path.join(img_dir, f"{count}.png")
        utils.plot_fig(
            final_outputs_flatten.max(),
            final_outputs_flatten.min(),
            original_im,
            pred,
            label,
            threshold,
            saved_path
        )
    
    return auroc


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Create checkpoint directory
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR,
        f"exp{len(os.listdir(const.CHECKPOINT_DIR))}_{args.category}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load configuration
    config = yaml.safe_load(open(args.config, "r"))

    # Build data loaders
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)

    # Setup TensorBoard
    log_cat_path = os.path.join(LOG_PATH, args.category)
    os.makedirs(log_cat_path, exist_ok=True)
    writer = SummaryWriter(log_cat_path)

    # Build model and optimizer
    print(f"Number of train epochs: {const.NUM_EPOCHS}")
    model = build_model(config, attention=args.attention_type)
    optimizer = build_optimizer(model)
    model.cuda()

    # Training loop
    max_auroc = 0
    max_epoch = 0
    max_model = None
    max_optimizer = None

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch, writer)
        
        if epoch % const.CHECKPOINT_INTERVAL == 0:
            model_auroc = eval_once(test_dataloader, model)
            writer.add_scalar('val auroc', model_auroc, epoch)

            if model_auroc > max_auroc:
                max_auroc = model_auroc
                print(f"*************** Current max AUROC: {epoch}_{max_auroc} ***************")
                max_epoch = epoch
                max_model = deepcopy(model.state_dict())
                max_optimizer = deepcopy(optimizer.state_dict())

    # Save best model
    torch.save(
        {
            "epoch": max_epoch,
            "model_state_dict": max_model,
            "optimizer_state_dict": max_optimizer,
        },
        os.path.join(checkpoint_dir, f"{max_epoch}_{max_auroc}.pt"),
    )
    
    print(f"*************** {args.category}: {max_epoch}_{max_auroc} ***************")
    writer.close()


def evaluate(args, plot_img=False):
    """
    Evaluate a trained model.
    
    Args:
        args: Command line arguments
        plot_img (bool): Whether to generate visualization images
    """
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config, attention=args.attention_type)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    
    if not plot_img:
        eval_once(test_dataloader, model)
    else:
        eval_test_images(test_dataloader, model, args.category)


def parse_args(cmd=None):
    """
    Parse command line arguments.
    
    Args:
        cmd (list): List of command line arguments (for programmatic use)
        
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train or evaluate on datasets")
    
    parser.add_argument(
        "-cfg", "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "-dataset", "--dataset_name",
        required=True,
        choices=["mvtec", "augmentedmvtec", "btad", "kltsdd", "visa"],
        help="Name of the dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset's folder"
    )
    parser.add_argument(
        "-cat", "--category",
        type=str,
        required=True,
        help="Category name in dataset"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run eval only"
    )
    parser.add_argument(
        "-ckpt", "--checkpoint",
        type=str,
        help="Path to load checkpoint"
    )
    parser.add_argument(
        "-attention", "--attention_type",
        default=None,
        choices=["eca", "gcnet", "shuffle", "triplet"],
        required=False,
        help="Type of attention"
    )
    
    if cmd:
        print(cmd)
        args = parser.parse_args(cmd)
    else:
        args = parser.parse_args()

    # Validate category
    if args.dataset_name == "augmentedmvtec":
        cat_choices_available = "MVTEC_CATEGORIES"
    else:
        cat_choices_available = args.dataset_name.upper() + "_CATEGORIES"

    if args.category not in getattr(const, cat_choices_available):
        parser.error("Your category should be on the list of categories of your dataset.")

    print(args)
    return args


if __name__ == "__main__":
     # Example training command
    cmd_train = [
        "-cfg=configs/resnet18.yaml",
        "-dataset=mvtec",
        "--data_path=datasets/MVTecAD",
        "-cat=cable",
        "-attention=eca"
    ]

    # Example evaluation command
    cmd_eval = [
        "-cfg=configs/deit.yaml",
        "-dataset=mvtec",
        "--data_path=datasets/MVTecAD",
        "-cat=toothbrush",
        "--eval",
        "-ckpt=/path/to/checkpoint.pt",
        "-attention=eca"
    ]

    # Categories to train on
    categories = [
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    ]

    # Set number of epochs
    const.NUM_EPOCHS = 100

    # Train on all categories
    for cat in categories:
        cmd_train[3] = f"-cat={cat}"
        args = parse_args(cmd_train)
        train(args)
    
    ## Eval
    # args = parse_args()
    # evaluate(args, plot_img=True)