import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from sklearn.model_selection import LeaveOneGroupOut

from .dataset import prepare_data_loaders, save_metadata
from .models import (create_concept_predictor, create_label_predictor,
                   create_end2end_model, create_joint_model)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_and_evaluate_fold(model, train_idx, val_idx, all_X, all_concepts, all_labels,
                            groups, device, num_epochs=30, learning_rate=0.001):
    LABEL_WEIGHT = 1.0
    CONCEPT_WEIGHT = 1.0

    concept_criterion = nn.CrossEntropyLoss()
    label_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_groups = groups[train_idx]
    train_unique_slices = np.unique(train_groups)

    loss_history = {'total': [], 'label': [], 'concept': []}

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        slice_features_list = []
        slice_labels_list = []
        slice_concept_losses = []

        for slice_name in train_unique_slices:
            mask = train_groups == slice_name
            local_indices = train_idx[mask]

            slice_X = torch.FloatTensor(all_X[local_indices]).to(device)
            slice_concepts = torch.LongTensor(all_concepts[local_indices]).to(device)
            slice_label = all_labels[local_indices[0]]

            concept_logits = model.concept_predictor(slice_X)
            concept_probs = torch.softmax(concept_logits, dim=1)
            slice_features = model.attention_aggregator(concept_probs.unsqueeze(0))

            slice_features_list.append(slice_features.squeeze(0))
            slice_labels_list.append(slice_label)
            slice_concept_losses.append(concept_criterion(concept_logits, slice_concepts))

        batch_features = torch.stack(slice_features_list, dim=0)
        batch_labels = torch.LongTensor(slice_labels_list).to(device)
        label_logits = model.label_predictor(batch_features)

        label_loss = label_criterion(label_logits, batch_labels)
        concept_loss = torch.stack(slice_concept_losses).mean()
        total_loss = LABEL_WEIGHT * label_loss + CONCEPT_WEIGHT * concept_loss

        total_loss.backward()
        optimizer.step()

        loss_history['total'].append(total_loss.item())
        loss_history['label'].append(label_loss.item())
        loss_history['concept'].append(concept_loss.item())

    model.eval()
    with torch.no_grad():
        val_X = torch.FloatTensor(all_X[val_idx]).to(device)
        val_concepts = torch.LongTensor(all_concepts[val_idx]).to(device)
        true_label = all_labels[val_idx[0]]

        concept_logits = model.concept_predictor(val_X)
        concept_probs = torch.softmax(concept_logits, dim=1)
        slice_features = model.attention_aggregator(concept_probs.unsqueeze(0))
        label_logits = model.label_predictor(slice_features)
        label_probs = torch.softmax(label_logits, dim=1)

        pred_label = label_logits.argmax(dim=1).item()
        concept_acc = accuracy(concept_logits, val_concepts, topk=(1,))[0].item()

        return {
            'true_label': true_label,
            'pred_label': pred_label,
            'label_probs': label_probs.cpu().numpy()[0],
            'concept_acc': concept_acc,
            'n_cells': len(val_idx),
            'loss_history': loss_history
        }


def run_loocv(all_X, all_concepts, all_labels, groups, metadata, device,
              num_epochs=5000, learning_rate=0.001, verbose=True, **model_kwargs):
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(all_X, all_labels, groups)

    loocv_results = []
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    all_concept_accs = []
    all_loss_histories = []

    if verbose:
        print(f"\nStarting LOOCV training ({n_splits} folds)...")
        print(f"  Training {num_epochs} epochs per fold")
        print(f"  Learning rate: {learning_rate}")
        print("  Using sklearn LeaveOneGroupOut\n")

    for fold_idx, (train_idx, val_idx) in enumerate(logo.split(all_X, all_labels, groups)):
        val_slice = groups[val_idx[0]]
        if verbose:
            print(f"[Fold {fold_idx+1}/{n_splits}] Validation slice: {val_slice}")
            print(f"  Train cells: {len(train_idx)}, Val cells: {len(val_idx)}")

        model = create_end2end_model(
            n_genes=metadata['n_genes'],
            n_concepts=metadata['n_concepts'],
            n_labels=metadata['n_labels'],
            concept_hidden_dims=model_kwargs.get('concept_hidden_dims', [512, 256]),
            label_hidden_dims=model_kwargs.get('label_hidden_dims', [256, 128]),
            concept_dropout=model_kwargs.get('concept_dropout', 0.3),
            label_dropout=model_kwargs.get('label_dropout', 0.2),
            use_sigmoid=model_kwargs.get('use_sigmoid', False),
            use_attention_aggregation=model_kwargs.get('use_attention_aggregation', True),
            attention_dim=model_kwargs.get('attention_dim', 64),
            use_multi_head=model_kwargs.get('use_multi_head', False),
            n_heads=model_kwargs.get('n_heads', 4),
        ).to(device)

        val_results = train_and_evaluate_fold(
            model=model,
            train_idx=train_idx,
            val_idx=val_idx,
            all_X=all_X,
            all_concepts=all_concepts,
            all_labels=all_labels,
            groups=groups,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )

        loocv_results.append({
            'fold': fold_idx + 1,
            'val_slice': val_slice,
            **val_results
        })
        all_true_labels.append(val_results['true_label'])
        all_pred_labels.append(val_results['pred_label'])
        all_pred_probs.append(val_results['label_probs'])
        all_concept_accs.append((val_results['concept_acc'], val_results['n_cells']))
        all_loss_histories.append(val_results['loss_history'])

        if verbose:
            correct = "✓" if val_results['true_label'] == val_results['pred_label'] else "✗"
            true_name = metadata['label_classes'][val_results['true_label']]
            pred_name = metadata['label_classes'][val_results['pred_label']]
            print(f"  True: {true_name}, Pred: {pred_name} {correct}")
            print(f"  Concept accuracy: {val_results['concept_acc']:.2f}%\n")

        del model
        torch.cuda.empty_cache()

    return {
        'loocv_results': loocv_results,
        'all_true_labels': np.array(all_true_labels),
        'all_pred_labels': np.array(all_pred_labels),
        'all_pred_probs': np.array(all_pred_probs),
        'all_concept_accs': all_concept_accs,
        'all_loss_histories': all_loss_histories,
    }
