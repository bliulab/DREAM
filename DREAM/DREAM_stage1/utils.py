import os
import ot
import warnings
import random
from anndata import AnnData
from scipy.sparse import spdiags
from matplotlib.pyplot import figure, imshow, axis, show
from matplotlib.image import imread
import pandas as pd
import numpy as np
import sklearn
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import torch.nn as nn
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
import scipy.sparse as sps
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from scipy.spatial.distance import *
from sklearn.metrics import *
from scipy.spatial import distance_matrix
from typing import List, Optional, Literal, Dict, Any
import anndata as ad
from scipy.sparse import csr_matrix

SEED = 2025
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    import torch_geometric
    torch_geometric.seed_everything(SEED)
except:
    pass

def get_class_attribute_names(img_dir = 'CUB_200_2011/images/', feature_file='CUB_200_2011/attributes/attributes.txt'):

    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split('.')[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)

    attr_id_to_name = dict()
    with open(feature_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ')
            attr_id_to_name[int(idx) - 1] = name
    return class_to_folder, attr_id_to_name

def sample_files(class_label, class_to_folder, number_of_files=10):

    folder = class_to_folder[class_label]
    class_files = random.sample(os.listdir(folder), number_of_files)
    class_files = [os.path.join(folder, f) for f in class_files]
    return class_files

def show_img_horizontally(list_of_files):

    fig = figure(figsize=(40,40))
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image)
        axis('off')
    show(block=True)

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))

    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))

    return data

def conctrust_data(args, adata, Concept_name=None):
    adata_X = adata.X
    gene_mat = torch.Tensor(adata_X)

    cell_coo = torch.Tensor(adata.obsm['spatial'])
    data = torch_geometric.data.Data(x=gene_mat, pos=cell_coo)
    data = torch_geometric.transforms.KNNGraph(k=args.k_graph, loop=True)(data)
    data.y = torch.Tensor(adata.obsm['pseudo_label'])

    if args.edge_weight:
        data = torch_geometric.transforms.Distance()(data)
        data.edge_weight = 1 - data.edge_attr[:,0]
    else:
        data.edge_weight = torch.ones(data.edge_index.size(1))

    if args.edge_weight:
        data = torch_geometric.transforms.Distance()(data)
        data.edge_weight = 1 - data.edge_attr[:,0]
    else:
        data.edge_weight = torch.ones(data.edge_index.size(1))

    from sklearn.preprocessing import LabelEncoder


    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(adata.obs['Region'])

    data.label = torch.tensor(encoded_labels, dtype=torch.long)

    attr_tensors = []
    attr_encoders = {}

    for concept in Concept_name:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(adata.obs[concept])
        print(f"{concept} category mapping:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))
        
        attr_tensors.append(torch.tensor(encoded, dtype=torch.long))
        attr_encoders[concept] = encoder 

    data.attr_labels = torch.stack(attr_tensors, dim=1) 

    return data


class KDLoss(nn.Module):

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, input, target):
        return nn.KLDivLoss()((input / self.T).log_softmax(1), (target / self.T).softmax(1)) * self.T * self.T
    

def batch_matrix_sqrt(Mats):

    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))


def calculate_covariance_matrices(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1, batch_size=None):

    
    ExpData = np.log(spatial_data[:, genes].X + 1)

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )
    
    global_mean = ExpData.mean(axis=0)
    n_cells = ExpData.shape[0]
    n_genes = len(genes)
    CovMats = np.zeros((n_cells, n_genes, n_genes), dtype=np.float32)
    if batch_size is None or batch_size >= n_cells:
        print("Calculating covariance matrices for all cells/spots")
        DistanceMatWeighted = (
            global_mean[None, None, :]
            - ExpData[kNNGraphIndex[np.arange(n_cells)]]
        )
        
        CovMats = np.matmul(
            DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted
        ) / (kNN - 1)
    else:
        batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
        for batch_idx in tqdm(batch_indices, desc="Calculating covariance matrices"):
            batch_neighbors = kNNGraphIndex[batch_idx]
            batch_distances = (
                global_mean[None, None, :]
                - ExpData[batch_neighbors]
            )
            batch_covs = np.matmul(
                batch_distances.transpose([0, 2, 1]), batch_distances
            ) / (kNN - 1)
            CovMats[batch_idx] = batch_covs
    reg_term = CovMats.mean() * 0.00001
    identity = np.eye(n_genes)[None, :, :]
    CovMats = CovMats + reg_term * identity
    
    return CovMats

def compute_covet(
    spatial_data, k=8, g=64, genes=None, spatial_key="spatial", batch_key="batch", batch_size=None
):


    genes = [] if genes is None else genes
    if batch_key not in spatial_data.obs.columns:
        batch_key = -1
    if g == -1 or g >= spatial_data.shape[1]:
        CovGenes = spatial_data.var_names
        print(f"Computing COVET using all {len(CovGenes)} genes")
    else:
        if "highly_variable" not in spatial_data.var.columns:
            print(f"Identifying top {g} highly variable genes for COVET calculation")
            spatial_data_copy = spatial_data.copy()
            if 'log' in spatial_data_copy.layers:
                layer = "log"
            elif 'log1p' in spatial_data_copy.layers:
                layer = "log1p"
            elif spatial_data_copy.X.min() < 0:
                layer = None
            else:
                spatial_data_copy.layers["log"] = np.log(spatial_data_copy.X + 1)
                layer = "log"
            sc.pp.highly_variable_genes(
                spatial_data_copy, 
                n_top_genes=g, 
                layer=layer if layer else None
            )
            hvg_genes = spatial_data_copy.var_names[spatial_data_copy.var.highly_variable]
            if(len(hvg_genes) > g):
                print(f"Fount {len(hvg_genes)} HVGs")
        else:
            hvg_genes = spatial_data.var_names[spatial_data.var.highly_variable]
            print(f"Using {len(hvg_genes)} pre-calculated highly variable genes for COVET")
        CovGenes = np.asarray(hvg_genes)
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)
            print(f"Added {len(genes)} user-specified genes to COVET calculation")
        print(f"Computing COVET using {len(CovGenes)} genes")
    COVET = calculate_covariance_matrices(
        spatial_data, k, genes=CovGenes, spatial_key=spatial_key, 
        batch_key=batch_key, batch_size=batch_size
    )
    if batch_size is None or batch_size >= COVET.shape[0]:
        print("Computing matrix square root...")
        COVET_SQRT = batch_matrix_sqrt(COVET)
    else:
        n_cells = COVET.shape[0]
        COVET_SQRT = np.zeros_like(COVET)
        batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
        for batch_idx in tqdm(batch_indices, desc="Computing matrix square roots"):
            batch_sqrt = batch_matrix_sqrt(COVET[batch_idx])
            COVET_SQRT[batch_idx] = batch_sqrt
    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes, dtype=str),
    )


def aggregate_mean(adj, x):
    return adj @ x
def aggregate_var(adj, x):
    mean = adj @ x
    mean_squared = adj @ (x * x)
    return mean_squared - mean * mean
def aggregate(adj, x, method):
    if method == "mean":
        return aggregate_mean(adj, x)
    elif method == "var":
        return aggregate_var(adj, x)
    else:
        raise NotImplementedError
def mul_broadcast(mat1, mat2):
    return spdiags(mat2, 0, len(mat2), len(mat2)) * mat1
def setdiag(array, value):
    if isinstance(array, sps.csr_matrix):
        array = array.tolil()
    array.setdiag(value)
    array = array.tocsr()
    if value == 0:
        array.eliminate_zeros()
    return array
def hop(adj_hop, adj, adj_visited=None):
    adj_hop = adj_hop @ adj

    if adj_visited is not None:
        adj_hop = adj_hop > adj_visited
        adj_visited = adj_visited + adj_hop

    return adj_hop, adj_visited
def normalize(adj):
    deg = np.array(np.sum(adj, axis=1)).squeeze()

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0

    return mul_broadcast(adj, deg_inv)
def _aggregate_neighbors(adj,X, nhood_layers, aggregations, disable_tqdm=True):
    adj = adj.astype(bool)
    adj = setdiag(adj, 0)
    adj_hop = adj.copy()
    adj_visited = setdiag(adj.copy(), 1)


    Xs = []
    for i in tqdm(range(0, max(nhood_layers) + 1), disable=disable_tqdm):
        if i in nhood_layers:
            if i == 0:
                Xs.append(X)
            else:
                if i > 1:
                    adj_hop, adj_visited = hop(adj_hop, adj, adj_visited)
                adj_hop_norm = normalize(adj_hop)

                for agg in aggregations:
                    x = aggregate(adj_hop_norm, X, agg)
                    Xs.append(x)
    if sps.issparse(X):
        return sps.hstack(Xs)
    else:
        return np.hstack(Xs)
    
def aggregate_neighbors(adata,n_layers=1,aggregations= "mean",connectivity_key=None,use_rep=None,sample_key=None,
                        out_key="X_cellcharter",copy=False):


    X = adata.X if use_rep is None else adata.obsm[use_rep]

    if isinstance(n_layers, int):
        n_layers = list(range(n_layers + 1))

    if sps.issparse(X):
        X_aggregated = sps.dok_matrix(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)), dtype=np.float32
        )
    else:
        X_aggregated = np.empty(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)), dtype=np.float32
        )

    if sample_key in adata.obs:
        samples = adata.obs[sample_key].unique()
        sample_idxs = [adata.obs[sample_key] == sample for sample in samples]
    else:
        sample_idxs = [np.arange(adata.shape[0])]

    for idxs in tqdm(sample_idxs, disable=(len(sample_idxs) == 1)):
        X_sample_aggregated = _aggregate_neighbors(
            adj=adata[idxs].obsp[connectivity_key],
            X=X[idxs],
            nhood_layers=n_layers,
            aggregations=aggregations,
            disable_tqdm=(len(sample_idxs) != 1),
        )
        X_aggregated[idxs] = X_sample_aggregated

    if isinstance(X_aggregated, sps.dok_matrix):
        X_aggregated = X_aggregated.tocsr()

    if copy:
        return X_aggregated

    adata.obsm[out_key] = X_aggregated



def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating intra-spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)


    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per spot on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    adata.uns['adj'] = G



def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2025):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):

    if adata.obsm['latent'].shape[1]>30:
        print('PCA_PCA_PCA')
        pca = PCA(n_components=20, random_state=42) 
        embedding = pca.fit_transform(adata.obsm['latent'].copy())
        adata.obsm['emb_pca'] = embedding
    else:
        adata.obsm['emb_pca'] = adata.obsm[key].copy()
    
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

def extract_top_value(map_matrix, retain_percent = 0.1):
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    
    return output 


  
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):

    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    


def remove_long_links(adata, distance_percentile = 90.0, connectivity_key = 'spatial_connectivities', 
                      distances_key = 'spatial_distances', neighs_key = 'spatial_neighbors', copy= False):

    conns, dists = adata.obsp[connectivity_key], adata.obsp[distances_key]

    if copy:
        conns, dists = conns.copy(), dists.copy()

    threshold = np.percentile(np.array(dists[dists != 0]).squeeze(), distance_percentile)
    conns[dists > threshold] = 0
    dists[dists > threshold] = 0

    conns.eliminate_zeros()
    dists.eliminate_zeros()

    if copy:
        return conns, dists
    else:
        adata.uns[neighs_key]["params"]["radius"] = threshold



def estimate_radius(adata):
    spatialmat = adata.obsm['spatial']
    min_dist_list = []
    cur_distmat = squareform(pdist(spatialmat))
    np.fill_diagonal(cur_distmat,np.inf)
    cur_min_dist = np.min(cur_distmat,axis=0)
    min_dist_list.append(cur_min_dist)
    min_dist_array = np.hstack(min_dist_list)
    neighbor_sz = np.median(min_dist_array)
    print(f'estimated radius: {neighbor_sz}')
    return neighbor_sz


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from typing import List, Optional, Literal, Dict, Any
import anndata as ad

def Cov_propress(
    adata: ad.AnnData,
    cat_covariates_keys: Optional[List[str]]=None,
    cat_covariates_no_edges: List[bool]=[],
    cat_covariates_cats: Optional[List[List]]=None,
    cat_covariates_embeds_nums: Optional[List[int]]=None):

    print('cat_covariates_keys:',cat_covariates_keys)
    if cat_covariates_keys is not None:
        for key in cat_covariates_keys:
            if key not in adata.obs.columns:
                raise ValueError(f"Categorical covariate '{key}' not found in adata.obs. Please check input.")

    if cat_covariates_cats is None:
        if cat_covariates_keys is not None:
            cat_covariates_cats_ = [
                adata.obs[key].unique().tolist() for key in cat_covariates_keys
            ]
        else:
            cat_covariates_cats_ = []
    else:
        cat_covariates_cats_ = cat_covariates_cats

    if cat_covariates_embeds_nums is None:
        cat_covariates_embeds_nums = []
        for cat_covariate_cats in cat_covariates_cats_:
            cat_covariates_embeds_nums.append(len(cat_covariate_cats))
    cat_covariates_embeds_nums_ = cat_covariates_embeds_nums

    if ((cat_covariates_no_edges is None) &
        (len(cat_covariates_cats_) > 0)):
        cat_covariates_no_edges_ = (
            [True] * len(cat_covariates_cats_))
    else:
        cat_covariates_no_edges_ = cat_covariates_no_edges

    if cat_covariates_keys is not None:
        for cat_covariate_key in cat_covariates_keys:
            if cat_covariate_key not in adata.obs:
                raise ValueError(
                    "Please specify adequate ´cat_covariates_keys´. "
                    f"The key {cat_covariate_key} was not found in adata.")
                

    return cat_covariates_cats_,cat_covariates_embeds_nums_,cat_covariates_no_edges_


from sklearn.preprocessing import LabelEncoder
import torch

def encode_cat_covariates(cat_covariates_cats: list[list[str]], device="cpu"):
    encoded_tensors = []
    encoders = []

    for cat_col in cat_covariates_cats:
        le = LabelEncoder()
        encoded = le.fit_transform(cat_col)
        tensor = torch.tensor(encoded, dtype=torch.long, device=device)
        encoded_tensors.append(tensor)
        encoders.append(le)

    return encoded_tensors, encoders


def encode_labels(adata: ad.AnnData,
                  label_encoder: dict,
                  label_key="str") -> np.ndarray:

    unique_labels = list(np.unique(adata.obs[label_key]))
    encoded_labels = np.zeros(adata.shape[0])

    if not set(unique_labels).issubset(set(label_encoder.keys())):
        print(f"Warning: Labels in adata.obs[{label_key}] are not a subset of "
              "the label encoder!")
        print("Therefore integer value of those labels is set to '-1'.")
        for unique_label in unique_labels:
            if unique_label not in label_encoder.keys():
                encoded_labels[adata.obs[label_key] == unique_label] = -1

    for label, label_encoding in label_encoder.items():
        encoded_labels[adata.obs[label_key] == label] = label_encoding
    return encoded_labels

def sparse_mx_to_sparse_tensor(sparse_mx: csr_matrix) -> SparseTensor:

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    torch_sparse_coo_tensor = torch.sparse.FloatTensor(indices, values, shape)
    sparse_tensor = SparseTensor.from_torch_sparse_coo_tensor(
        torch_sparse_coo_tensor)
    return sparse_tensor

class SpatialAnnTorchDataset():
    def __init__(self,
                 adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 edge_label_adj_key: str="edge_label_spatial_connectivities",
                 self_loops: bool=True,
                 cat_covariates_keys: Optional[str]=None):

        x = adata.X
        if sp.issparse(x): 
            self.x = torch.tensor(x.toarray())
        else:
            self.x = torch.tensor(x)
        if adata_atac is not None:
            if sp.issparse(adata_atac.X): 
                self.x = torch.cat(
                    (self.x, torch.tensor(adata_atac.X.toarray())), axis=1)
            else:
                self.x = torch.cat((self.x, torch.tensor(adata_atac.X)), axis=1)
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))
        if edge_label_adj_key in adata.obsp:
            self.edge_label_adj = sp.csr_matrix(adata.obsp[edge_label_adj_key])
        else:
            self.edge_label_adj = None
        if (self.adj.nnz() != self.adj.t().nnz()):
            raise ImportError("The input adjacency matrix has to be symmetric.")
        
        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()

        if self_loops:
            self.edge_index, _ = remove_self_loops(self.edge_index)
            self.edge_index, _ = add_self_loops(self.edge_index,
                                                num_nodes=self.x.size(0))
            
        if cat_covariates_keys is not None:
            self.cat_covariates_cats = []
            for cat_covariate_key, cat_covariate_label_encoder in zip(
                cat_covariates_keys,
                cat_covariates_label_encoders):
                cat_covariate_cats = torch.tensor(
                    encode_labels(adata,
                                  cat_covariate_label_encoder,
                                  cat_covariate_key), dtype=torch.long)
                self.cat_covariates_cats.append(cat_covariate_cats)
            self.cat_covariates_cats = torch.stack(self.cat_covariates_cats,
                                                   dim=1)            

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1)
    def __len__(self):
        return self.x.size(0)