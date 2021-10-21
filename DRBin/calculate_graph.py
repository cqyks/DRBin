import torch
import numpy as np


def smaller_indices(distances, threshold):
    arr = distances.numpy()
    indices = (arr <= threshold).nonzero()[0]
    torch_indices = torch.from_numpy(indices)
    #print(type(indices), type(torch_indices))
    return torch_indices

def calc_distances(matrix, index):
    "Return vecembeddedRoottor of cosine distances from rows of normalized matrix to given row."
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0  # avoid float rounding errors
    return dists

def normalize(matrix, inplace=False):
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    matrix = matrix.clone()

    # If any rows are kept all zeros, the distance function will return 0.5 to all points
    # inclusive itself, which can break the code in this module
    zeromask = matrix.sum(dim=1) == 0
    matrix[zeromask] = 1/matrix.shape[1]
    matrix /= (matrix.norm(dim=1).reshape(-1, 1) * (2 ** 0.5))
    return matrix

def calculate_graph(latent):
    a = []
    b = []
    latent = normalize(latent)
    for i in range(len(latent)):
        dist = calc_distances(latent, i)
        edges = smaller_indices(dist, 0.005)
        for e in edges:
            a.append(i)
            b.append(e)
    
    return a, b