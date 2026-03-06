import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.3, use_bn=True):
        super(MLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = []

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConceptPredictor(nn.Module):
    def __init__(self, n_genes, n_concepts, hidden_dims=[512, 256], dropout=0.3):
        super(ConceptPredictor, self).__init__()
        self.mlp = MLP(n_genes, n_concepts, hidden_dims, dropout)

    def forward(self, x):
        return self.mlp(x)


class AttentionAggregation(nn.Module):
    def __init__(self, n_concepts, attention_dim=64, use_multi_head=False, n_heads=4):
        super(AttentionAggregation, self).__init__()
        self.n_concepts = n_concepts
        self.use_multi_head = use_multi_head

        if use_multi_head:
            self.attention = nn.MultiheadAttention(
                embed_dim=n_concepts,
                num_heads=n_heads,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.query = nn.Linear(n_concepts, attention_dim)
            self.key = nn.Linear(n_concepts, attention_dim)
            self.value = nn.Linear(n_concepts, attention_dim)
            self.out = nn.Linear(attention_dim, n_concepts)

        self.scale = np.sqrt(attention_dim)

    def forward(self, concept_probs, slice_mask=None):
        if self.use_multi_head:
            query = concept_probs.mean(dim=1, keepdim=True)

            attn_output, attn_weights = self.attention(
                query, concept_probs, concept_probs,
                key_padding_mask=slice_mask if slice_mask is not None else None
            )

            weighted = attn_output.squeeze(1)
        else:
            Q = self.query(concept_probs)
            K = self.key(concept_probs)
            V = self.value(concept_probs)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            if slice_mask is not None:
                scores = scores.masked_fill(slice_mask.unsqueeze(1) == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)

            weighted_V = torch.matmul(attn_weights, V)
            weighted_V = weighted_V.mean(dim=1)

            weighted = self.out(weighted_V)

        mean_pooling = concept_probs.mean(dim=1)
        max_pooling = concept_probs.max(dim=1)[0]
        std_pooling = concept_probs.std(dim=1)

        aggregated = torch.cat([
            weighted,
            mean_pooling,
            max_pooling,
            std_pooling
        ], dim=-1)

        return aggregated


class LabelPredictor(nn.Module):
    def __init__(self, n_concepts, n_labels, hidden_dims=[128, 64], dropout=0.2,
                 use_attention=False, attention_dim=64):
        super(LabelPredictor, self).__init__()
        self.use_attention = use_attention

        if use_attention:
            input_dim = 4 * n_concepts
        else:
            input_dim = n_concepts

        self.mlp = MLP(input_dim, n_labels, hidden_dims, dropout)

    def forward(self, x):
        return self.mlp(x)


class End2EndModel(nn.Module):
    def __init__(self, concept_predictor, label_predictor, use_sigmoid=False,
                 attention_aggregator=None):
        super(End2EndModel, self).__init__()
        self.concept_predictor = concept_predictor
        self.label_predictor = label_predictor
        self.use_sigmoid = use_sigmoid
        self.attention_aggregator = attention_aggregator

    def forward(self, x, return_concepts=False, cell_level=True):
        if cell_level:
            concept_logits = self.concept_predictor(x)

            if self.use_sigmoid:
                concept_probs = torch.sigmoid(concept_logits)
            else:
                concept_probs = F.softmax(concept_logits, dim=1)

            label_logits = self.label_predictor(concept_probs)

            if return_concepts:
                return label_logits, concept_logits
            else:
                return label_logits
        else:
            label_logits = self.label_predictor(x)
            return label_logits

    def predict_concepts(self, x):
        concept_logits = self.concept_predictor(x)

        if self.use_sigmoid:
            concept_probs = torch.sigmoid(concept_logits)
        else:
            concept_probs = F.softmax(concept_logits, dim=1)

        return concept_probs

    def predict_labels(self, concept_dist):
        return self.label_predictor(concept_dist)

    def aggregate_cells_to_slice(self, cell_concept_probs, slice_ids=None):
        if self.attention_aggregator is None:
            raise ValueError("Attention aggregator not initialized. Please pass attention_aggregator when creating the model.")

        if slice_ids is None:
            cell_concept_probs = cell_concept_probs.unsqueeze(0)
            slice_features = self.attention_aggregator(cell_concept_probs)
        else:
            unique_slices = torch.unique(slice_ids)
            slice_features_list = []

            for slice_id in unique_slices:
                mask = slice_ids == slice_id
                slice_cells = cell_concept_probs[mask]
                slice_cells = slice_cells.unsqueeze(0)

                slice_feat = self.attention_aggregator(slice_cells)
                slice_features_list.append(slice_feat)

            slice_features = torch.cat(slice_features_list, dim=0)

        return slice_features


class JointModel(nn.Module):
    def __init__(self, n_genes, n_concepts, n_labels,
                 concept_hidden_dims=[512, 256],
                 label_hidden_dims=[512, 256],
                 dropout=0.3):
        super(JointModel, self).__init__()

        self.feature_extractor = MLP(n_genes, concept_hidden_dims[-1],
                                     concept_hidden_dims[:-1], dropout)

        self.concept_head = nn.Linear(concept_hidden_dims[-1], n_concepts)

        self.label_head = MLP(concept_hidden_dims[-1], n_labels,
                             label_hidden_dims, dropout)

    def forward(self, x):
        features = self.feature_extractor(x)
        concept_logits = self.concept_head(features)
        label_logits = self.label_head(features)
        return label_logits, concept_logits


def create_concept_predictor(n_genes, n_concepts, hidden_dims=[512, 256], dropout=0.3):
    return ConceptPredictor(n_genes, n_concepts, hidden_dims, dropout)


def create_label_predictor(n_concepts, n_labels, hidden_dims=[128, 64], dropout=0.2,
                          use_attention=False, attention_dim=64):
    return LabelPredictor(n_concepts, n_labels, hidden_dims, dropout,
                         use_attention, attention_dim)


def create_end2end_model(n_genes, n_concepts, n_labels,
                         concept_hidden_dims=[512, 256],
                         label_hidden_dims=[128, 64],
                         concept_dropout=0.3,
                         label_dropout=0.2,
                         use_sigmoid=False,
                         use_attention_aggregation=False,
                         attention_dim=64,
                         use_multi_head=False,
                         n_heads=4):
    concept_predictor = create_concept_predictor(n_genes, n_concepts,
                                                 concept_hidden_dims, concept_dropout)

    label_predictor = create_label_predictor(
        n_concepts, n_labels, label_hidden_dims, label_dropout,
        use_attention=use_attention_aggregation, attention_dim=attention_dim
    )

    attention_aggregator = None
    if use_attention_aggregation:
        attention_aggregator = AttentionAggregation(
            n_concepts, attention_dim, use_multi_head, n_heads
        )

    return End2EndModel(concept_predictor, label_predictor, use_sigmoid, attention_aggregator)


def create_joint_model(n_genes, n_concepts, n_labels,
                      concept_hidden_dims=[512, 256],
                      label_hidden_dims=[512, 256],
                      dropout=0.3):
    return JointModel(n_genes, n_concepts, n_labels,
                     concept_hidden_dims, label_hidden_dims, dropout)
