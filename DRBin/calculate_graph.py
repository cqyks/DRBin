import torch
import numpy as np
import collections

def smaller_indices(distances, threshold):
    arr = distances.numpy()
    indices = (arr <= threshold).nonzero()[0]
    torch_indices = torch.from_numpy(indices)
    #print(type(indices), type(torch_indices))
    return torch_indices

def larger_indices(distances, threshold):
    arr = distances.numpy()
    indices = (arr >= threshold).nonzero()[0]
    torch_indices = torch.from_numpy(indices)
    #print(type(indices), type(torch_indices))
    return torch_indices


def calc_distances(matrix, index):
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0
    return dists

def normalize(matrix, inplace=False):
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    matrix = matrix.clone()

    zeromask = matrix.sum(dim=1) == 0
    matrix[zeromask] = 1/matrix.shape[1]
    matrix /= (matrix.norm(dim=1).reshape(-1, 1) * (2 ** 0.5))
    return matrix

def phrase(marker_contigs):
    contig_relation = {}
    for marker, contigs in marker_contigs.items():
        for contig in contigs:
            if contig.find('[') == -1:
                continue
            contig_relation[contig] = contig[contig.find('[') + 1:contig.find(']')]
    return contig_relation

def calculate_graph(latent, marker_contigs, contig_id_idx):
    a = []
    b = []
    edge = dict()
    edge = collections.defaultdict(set)
    latent = normalize(latent)
    for i in range(len(latent)):
        dist = calc_distances(latent, i)
        edges = smaller_indices(dist, 0.0005)
        for e in edges:
            edge[i].add(e)
    contig_relation = phrase(marker_contigs)
    
    for contig1 in contig_relation.keys():
        for contig2 in contig_relation.keys():
            r = contig_id_idx[contig1]
            s = contig_id_idx[contig2]
            edge[r].add(s)
    
    for k, v in edge.items():
        for s in v:
            a.append(k)
            b.append(s)
    
    return a, b

def calculate_negativate_graph(latent, marker_contigs, contig_id_idx):
    a = []
    b = []
    edge = dict()
    edge = collections.defaultdict(set)
    latent = normalize(latent)
    for i in range(len(latent)):
        dist = calc_distances(latent, i)
        edges = larger_indices(dist, 0.75)
        for e in edges:
            edge[i].add(e)
    contig_relation = phrase(marker_contigs)
    
    for contig1 in contig_relation.keys():
        for contig2 in contig_relation.keys():
            r = contig_id_idx[contig1]
            s = contig_id_idx[contig2]
            edge[r].add(s)
            
    for k, v in edge.items():
        for s in v:
            a.append(k)
            b.append(s)
    
    return a, b
