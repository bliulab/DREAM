import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class SpatialDataset(Dataset):
    def __init__(self, X, concepts, labels):
        self.X = torch.FloatTensor(X)
        self.concepts = torch.LongTensor(concepts)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.concepts[idx], self.labels[idx]


class SliceLevelDataset(Dataset):
    def __init__(self, slice_features, slice_labels):
        self.features = torch.FloatTensor(slice_features)
        self.labels = torch.LongTensor(slice_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_spatial_data(data_path, concept_key='leiden', label_key='type', slice_key='sample', expr_key='X_raw'):
    adata = sc.read_h5ad(data_path)

    if expr_key in adata.obsm:
        X = adata.obsm[expr_key]
    else:
        raise ValueError(f"expr_key '{expr_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    if hasattr(X, 'toarray'):
        X = X.toarray()
    else:
        X = np.array(X)

    concepts = adata.obs[concept_key].values
    concept_encoder = LabelEncoder()
    concepts_encoded = concept_encoder.fit_transform(concepts)

    labels = adata.obs[label_key].values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    slice_ids = adata.obs[slice_key].values

    return X, concepts_encoded, labels_encoded, slice_ids, concept_encoder, label_encoder


def prepare_data_loaders(data_path,
                         val_split=0.2, seed=42,
                         concept_key='leiden', label_key='type',
                         slice_key='slice_name', expr_key='X_raw',
                         use_all_data=False):
    X, concepts, labels, slice_ids, concept_encoder, label_encoder = load_spatial_data(
        data_path, concept_key, label_key, slice_key, expr_key
    )

    unique_slices = np.unique(slice_ids)

    if use_all_data:
        train_slices = unique_slices
        val_slices = unique_slices
    else:
        train_slices, val_slices = train_test_split(
            unique_slices, test_size=val_split, random_state=seed
        )

    train_mask = np.isin(slice_ids, train_slices)
    val_mask = np.isin(slice_ids, val_slices)

    train_dataset = SpatialDataset(X[train_mask], concepts[train_mask], labels[train_mask])
    val_dataset = SpatialDataset(X[val_mask], concepts[val_mask], labels[val_mask])
    all_dataset = SpatialDataset(X, concepts, labels)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    all_loader = DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False)

    metadata = {
        'concept_encoder': concept_encoder,
        'label_encoder': label_encoder,
        'n_genes': X.shape[1],
        'n_concepts': len(concept_encoder.classes_),
        'n_labels': len(label_encoder.classes_),
        'train_slices': train_slices,
        'val_slices': val_slices,
        'all_slices': unique_slices,
        'concept_classes': concept_encoder.classes_,
        'label_classes': label_encoder.classes_,
    }

    return (train_loader, val_loader, all_loader, metadata)


def save_metadata(metadata, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_metadata(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)
