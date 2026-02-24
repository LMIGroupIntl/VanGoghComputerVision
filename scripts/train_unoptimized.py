#!/usr/bin/env python3
"""
Multiple-Patch Model Training Script
This script implements uniform patching, patch-level training with sample weights, and painting-level aggregation.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
import timm
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit
import torch.nn.utils as U
from collections import defaultdict, Counter
import json
from datetime import datetime
from timm.scheduler import CosineLRScheduler
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)

# Configuration
DEFAULT_CONFIG = {
    'target_size': 256,
    'imitation_weight': 1.0,
    'batch_size': 16,
    'epochs': 30,
    'n_freeze_epochs_swin': 5,
    'patience_reduce_lr': 5,
    'swin_lr_head_frozen': 3e-5,
    'swin_lr_backbone_frozen': 0.0,
    'swin_lr_head_finetune': 8e-6,
    'swin_lr_backbone_finetune': 3e-6,
    'swin_lr_min_finetune': 1e-6,
    'effnet_lr_initial': 1e-5,
    'weight_decay': 1e-2,
    'output_dir': './output',
    'print_every': 50
}

class MultiPatchDataset(Dataset):
    def __init__(self, root_dir, target_size=256, imitation_weight=1.0):
        self.root_dir = root_dir
        self.target_size = target_size
        self.imitation_weight = imitation_weight
        self.defined_classes = ['authentic', 'imitation']  # authentic=0, imitation=1
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.defined_classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Dataset initialized. Root: '{self.root_dir}', Class mapping: {self.class_to_idx}")

        self.all_patches = []  # Will store (PIL.Image, label, painting_id)
        self._prepare_data()

        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _prepare_data(self):
        painting_id_counter = 0
        for class_name in self.defined_classes:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"Warning: Class directory not found: {class_path}")
                continue

            label = self.class_to_idx[class_name]
            
            image_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not image_files:
                print(f"Warning: No image files found in {class_path}")
                continue

            for img_filename in image_files:
                img_path = os.path.join(class_path, img_filename)
                current_painting_id = painting_id_counter

                try:
                    img = Image.open(img_path).convert('RGB')
                    w, h = img.size
                    max_dim = max(w, h)

                    if max_dim == 0:
                        print(f"Warning: Image {img_path} has zero dimension, skipping.")
                        continue

                    if max_dim > 1024:
                        grid_size = 4  # 4x4 patches
                    elif max_dim >= 512:
                        grid_size = 2  # 2x2 patches
                    else:
                        grid_size = 1  # 1x1 patch

                    patch_width = w // grid_size
                    patch_height = h // grid_size

                    if patch_width == 0 or patch_height == 0:
                        print(f"Warning: Calculated patch size is zero for {img_path} (w={w},h={h},grid={grid_size}), skipping.")
                        continue

                    for i in range(grid_size):
                        for j in range(grid_size):
                            left = j * patch_width
                            upper = i * patch_height
                            right = (j + 1) * patch_width if (j + 1) < grid_size else w
                            bottom = (i + 1) * patch_height if (i + 1) < grid_size else h
                            
                            patch_img = img.crop((left, upper, right, bottom))
                            if patch_img.size[0] > 0 and patch_img.size[1] > 0:
                                self.all_patches.append((patch_img, label, current_painting_id))
                            else:
                                print(f"Warning: Generated empty patch for {img_path} at grid ({i},{j}), skipping.")
                    
                    painting_id_counter += 1

                except UnidentifiedImageError:
                    print(f"Warning: Cannot identify image file {img_path}, skipping.")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
        
        if not self.all_patches:
            print(f"Warning: No patches were generated. Check dataset structure in '{self.root_dir}', paths, and image files.")
        else:
            print(f"Generated {len(self.all_patches)} patches from {painting_id_counter} paintings.")

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        patch_img, label, painting_id = self.all_patches[idx]
        
        try:
            transformed_patch = self.transform(patch_img)
        except Exception as e:
            print(f"Error transforming patch (original index {idx}, painting_id {painting_id}): {e}")
            transformed_patch = torch.zeros((3, self.target_size, self.target_size), dtype=torch.float32)
            
        return transformed_patch, label, painting_id

def setup_device():
    """Setup and return the appropriate device."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def create_data_loaders(dataset, config):
    """Create train and validation data loaders."""
    groups = [pid for _, _, pid in dataset.all_patches]
    
    # Split with GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(dataset.all_patches, groups=groups))
    
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Classes: {list(dataset.class_to_idx.keys())}")
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    # Sample weights for imbalance
    train_labels_for_sampler = np.array([dataset.all_patches[i][1] for i in train_idx])
    weights = [dataset.imitation_weight if label == dataset.class_to_idx.get('imitation', 1) else 1 
               for label in train_labels_for_sampler]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, train_idx, val_idx

def calculate_class_weights(dataset, train_idx, device):
    """Calculate class weights for loss function."""
    train_labels_for_class_weights = [dataset.all_patches[i][1] for i in train_idx]
    class_counts = Counter(train_labels_for_class_weights)
    num_classes = len(dataset.class_to_idx)
    
    weights_list = [0.0] * num_classes
    
    for class_idx, count in class_counts.items():
        if count > 0:
            weights_list[class_idx] = len(train_labels_for_class_weights) / (num_classes * count)
        else:
            weights_list[class_idx] = 1.0
    
    class_weights = torch.tensor(weights_list, dtype=torch.float).to(device)
    print(f"Calculated class weights: {class_weights.tolist()}")
    return class_weights

def setup_models_and_optimizers(config, device):
    """Setup models and optimizers."""
    print("--- Setting up models and optimizers ---")
    
    models = {}
    models['efficientnet'] = timm.create_model('efficientnet_b5', pretrained=True, num_classes=2).to(device)
    models['swin'] = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, 
                                       num_classes=2, img_size=config['target_size']).to(device)
    
    # EfficientNet optimizer
    eff_optimizer = torch.optim.AdamW(models['efficientnet'].parameters(), 
                                      lr=config['effnet_lr_initial'], 
                                      weight_decay=config['weight_decay'])
    
    # Swin optimizer with differential learning rates
    swin_model_instance = models['swin']
    
    try:
        head_params_swin = list(swin_model_instance.head.parameters())
        head_param_ids_swin = {id(p) for p in head_params_swin}
        backbone_params_swin = [p for p in swin_model_instance.parameters() 
                               if id(p) not in head_param_ids_swin]
        
        if not backbone_params_swin:
            print("Warning: Swin backbone parameters list is empty. Falling back.")
            for param in swin_model_instance.parameters():
                param.requires_grad = True
            swin_optimizer = torch.optim.AdamW(swin_model_instance.parameters(), 
                                              lr=config['swin_lr_head_frozen'], 
                                              weight_decay=config['weight_decay'])
        else:
            # Freeze backbone layers initially
            for param in backbone_params_swin:
                param.requires_grad = False
            for param in head_params_swin:
                param.requires_grad = True
                
            swin_optimizer = torch.optim.AdamW([
                {'params': backbone_params_swin, 'lr': config['swin_lr_backbone_frozen'], 'name': 'swin_backbone'},
                {'params': head_params_swin, 'lr': config['swin_lr_head_frozen'], 'name': 'swin_head'}
            ], weight_decay=config['weight_decay'])
            print("Swin Transformer Optimizer (Frozen Backbone Phase - DLR) configured.")
            
    except AttributeError:
        print("Error: swin_model_instance.head not found. Falling back to single learning rate.")
        for param in swin_model_instance.parameters():
            param.requires_grad = True
        swin_optimizer = torch.optim.AdamW(swin_model_instance.parameters(), 
                                          lr=config['swin_lr_head_frozen'], 
                                          weight_decay=config['weight_decay'])
    
    optimizers = {
        'efficientnet': eff_optimizer,
        'swin': swin_optimizer
    }
    
    return models, optimizers

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch=None, print_every=10):
    """Train model for one epoch."""
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch_idx, (x, y, _) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        # Gradient clipping
        if hasattr(model, 'default_cfg') and 'swin' in model.default_cfg.get('architecture', '').lower():
            grad_norm = U.clip_grad_norm_(model.parameters(), max_norm=5.0)
        else:
            grad_norm = U.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * x.size(0)

        preds = out.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().tolist())

        if epoch is not None and batch_idx % print_every == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss {loss.item():.6f}")

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds

def eval_dataset_with_loss(model, loader, loss_fn, device):
    """Evaluate dataset with painting-level aggregation."""
    model.eval()
    group_logits = defaultdict(list)
    group_labels = {}
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y, pid in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            batch_loss = loss_fn(logits, y)
            total_loss += batch_loss.item() * x.size(0)
            total_samples += x.size(0)
            
            logits_cpu = logits.cpu()
            for lg, yy, id_ in zip(logits_cpu, y.cpu(), pid):
                group_logits[id_].append(lg)
                group_labels[id_] = int(yy)

    avg_loss = total_loss / total_samples
    
    y_true = list(group_labels.values())
    y_pred = [int(torch.stack(lgs).mean(0).argmax()) for lgs in group_logits.values()]
    acc = sum(yt==yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    
    return acc, avg_loss

def save_training_results(model_name, train_metrics, val_metrics, timestamp, output_dir):
    """Save detailed training results to JSON file."""
    results = {
        "model_name": model_name,
        "timestamp": timestamp,
        "training": train_metrics,
        "validation": val_metrics
    }
    json_path = os.path.join(output_dir, f"{model_name}_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Detailed results saved to {json_path}")

def train_models(models, optimizers, train_loader, val_loader, loss_fn, config, device, output_dir):
    """Main training loop."""
    # Setup schedulers
    schedulers = {}
    for name in models.keys():
        if name == 'swin':
            schedulers[name] = None  # Will be initialized later
        else:
            schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizers[name], mode='min', factor=0.5, 
                patience=config['patience_reduce_lr']
            )
    
    # Metrics tracking
    train_metrics = {name: [] for name in models.keys()}
    val_metrics = {name: [] for name in models.keys()}
    val_loss_metrics = {name: [] for name in models.keys()}
    
    swin_is_unfrozen = False
    swin_finetune_scheduler_step_count = 0
    
    print(f"\n--- Starting Training for {config['epochs']} Epochs ---")
    
    for epoch in range(config['epochs']):
        print(f"\n===== EPOCH {epoch + 1}/{config['epochs']} =====")
        
        # Unfreeze Swin backbone if needed
        if 'swin' in models and epoch == config['n_freeze_epochs_swin'] and not swin_is_unfrozen:
            print(f"\n-- Epoch {epoch + 1}: Unfreezing Swin Transformer backbone --")
            
            swin_model = models['swin']
            swin_optimizer = optimizers['swin']
            
            # Set all parameters to trainable
            for param in swin_model.parameters():
                param.requires_grad = True
            
            # Adjust learning rates
            if swin_optimizer:
                for param_group in swin_optimizer.param_groups:
                    for p in param_group['params']:
                        p.requires_grad = True
                    
                    group_name = param_group.get('name', 'unknown_group')
                    if group_name == 'swin_backbone':
                        param_group['lr'] = config['swin_lr_backbone_finetune']
                    elif group_name == 'swin_head':
                        param_group['lr'] = config['swin_lr_head_finetune']
                
                # Initialize CosineLR scheduler
                remaining_epochs = config['epochs'] - epoch
                schedulers['swin'] = CosineLRScheduler(
                    swin_optimizer,
                    t_initial=remaining_epochs,
                    warmup_t=1,
                    warmup_lr_init=config['swin_lr_backbone_finetune'],
                    lr_min=config['swin_lr_min_finetune']
                )
                swin_is_unfrozen = True
                swin_finetune_scheduler_step_count = 0
        
        # Train each model
        for model_name in models.keys():
            model = models[model_name]
            optimizer = optimizers[model_name]
            scheduler = schedulers.get(model_name)
            
            print(f"\nTraining {model_name} for Epoch {epoch + 1}...")
            
            train_loss, _, _ = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, 
                epoch, config['print_every']
            )
            
            print(f"Validating {model_name} for Epoch {epoch + 1}...")
            val_acc, val_loss = eval_dataset_with_loss(model, val_loader, loss_fn, device)
            
            # Scheduler step
            if scheduler:
                if model_name == 'swin' and swin_is_unfrozen:
                    scheduler.step(swin_finetune_scheduler_step_count)
                elif model_name != 'swin':
                    scheduler.step(train_loss)
            
            # Log results
            log_message = f"RESULTS: {model_name} | Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.6f} | Val Acc: {val_acc:.6f}"
            print(log_message)
            
            train_metrics[model_name].append(train_loss)
            val_metrics[model_name].append(val_acc)
            val_loss_metrics[model_name].append(val_loss)
        
        if swin_is_unfrozen:
            swin_finetune_scheduler_step_count += 1
    
    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for model_name in models.keys():
        save_training_results(model_name, train_metrics[model_name], 
                            val_metrics[model_name], timestamp, output_dir)
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_name}_final_{timestamp}.pth")
        torch.save(models[model_name].state_dict(), model_path)
        print(f"Saved {model_name} model to {model_path}")
    
    return models, train_metrics, val_metrics

def evaluate_models(models, val_loader, device, output_dir):
    """Detailed evaluation with metrics and curves."""
    print("\n--- Detailed Evaluation ---")
    
    for name, model in models.items():
        model.eval()
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)[:, 1]  # P(imitation)
                y_true.extend(y.cpu().numpy().tolist())
                y_scores.extend(probs.cpu().numpy().tolist())
        
        y_pred = [1 if p >= 0.5 else 0 for p in y_scores]
        
        # Compute metrics
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        print(f"\n=== {name.upper()} ===")
        print(f"ROC-AUC: {auc:.3f}   PR-AUC (AP): {ap:.3f}")
        print(classification_report(y_true, y_pred, target_names=['authentic', 'imitation']))
        
        # Save plots
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        plt.title(f'ROC Curve – {name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{name}_roc_curve.png'))
        plt.close()
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.title(f'PR Curve – {name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{name}_pr_curve.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train multiple patch classification models')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to dataset directory (should contain authentic/ and imitation/ subdirs)')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for models and results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--target_size', type=int, default=256,
                       help='Target image size for patches')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'target_size': args.target_size,
        'output_dir': args.output_dir
    })
    
    print("Starting model training...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup device
    device = setup_device()
    
    # Create dataset and data loaders
    dataset = MultiPatchDataset(args.data_dir, config['target_size'], config['imitation_weight'])
    train_loader, val_loader, train_idx, val_idx = create_data_loaders(dataset, config)
    
    # Calculate class weights
    class_weights = calculate_class_weights(dataset, train_idx, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Setup models and optimizers
    models, optimizers = setup_models_and_optimizers(config, device)
    
    # Train models
    trained_models, train_metrics, val_metrics = train_models(
        models, optimizers, train_loader, val_loader, loss_fn, config, device, args.output_dir
    )
    
    # Evaluate models
    evaluate_models(trained_models, val_loader, device, args.output_dir)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()