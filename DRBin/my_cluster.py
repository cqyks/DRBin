from typing import Counter
import numpy as np
from collections import defaultdict, Counter
from collections import deque
import torch
import matplotlib.pyplot as plt
import math
import random
import pickle
import os
from Bio import SeqIO
import shutil
import random

EPS = 1e-3
MAXSTEPS = 25
MINSUCCESSES = 20

def smaller_indices(distances, threshold):
    arr = distances.numpy()
    indices = (arr <= threshold).nonzero()[0]
    torch_indices = torch.from_numpy(indices)
    #print(type(indices), type(torch_indices))
    return torch_indices


def normalize(matrix, inplace=False):
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    matrix = matrix.clone()
    zeromask = matrix.sum(dim=1) == 0
    matrix[zeromask] = 1/matrix.shape[1]
    matrix /= (matrix.norm(dim=1).reshape(-1, 1) * (2 ** 0.5))
    return matrix


def calc_distances(matrix, index):
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0  # avoid float rounding errors
    return dists


_DELTA_X = 0.005
_XMAX = 0.3
_DEFAULT_RADIUS = 0.06
_MEDOID_RADIUS = 0.05

# This is the PDF of normal with µ=0, s=0.01 from -0.075 to 0.075 with intervals
# of DELTA_X, for a total of 31 values. We multiply by _DELTA_X so the density
# of one point sums to approximately one
_NORMALPDF = _DELTA_X * torch.Tensor(
    [2.43432053e-11, 9.13472041e-10, 2.66955661e-08, 6.07588285e-07,
     1.07697600e-05, 1.48671951e-04, 1.59837411e-03, 1.33830226e-02,
     8.72682695e-02, 4.43184841e-01, 1.75283005e+00, 5.39909665e+00,
     1.29517596e+01, 2.41970725e+01, 3.52065327e+01, 3.98942280e+01,
     3.52065327e+01, 2.41970725e+01, 1.29517596e+01, 5.39909665e+00,
     1.75283005e+00, 4.43184841e-01, 8.72682695e-02, 1.33830226e-02,
     1.59837411e-03, 1.48671951e-04, 1.07697600e-05, 6.07588285e-07,
     2.66955661e-08, 9.13472041e-10, 2.43432053e-11])


def calc_densities(histogram, pdf=_NORMALPDF):
    pdf_len = len(pdf)

    densities = torch.zeros(len(histogram) + pdf_len - 1)
    for i in range(len(densities) - pdf_len + 1):
        densities[i:i+pdf_len] += pdf * histogram[i]

    densities = densities[15:-15]

    return densities

def sample_medoid(matrix, medoid, threshold):

    distances = calc_distances(matrix, medoid)
    cluster = smaller_indices(distances, threshold)

    if len(cluster) == 1:
        average_distance = 0.0
    else:
        average_distance = distances[cluster].sum().item() / (len(cluster) - 1)

    return cluster, distances, average_distance

#code from vamb
def find_valley_ratio(histogram, peak_valley_ratio):
    peak_density = 0
    min_density = None
    peak_over = False
    success = False

    minima = None
    maxima = None
    early_minima = None
    delta_x = _XMAX / len(histogram)
    densities = calc_densities(histogram)
    x = 0
    
    if histogram[:10].sum().item() == 0:
        return None, 0.025, 0.025
    
    for n, density in enumerate(densities):
        if not peak_over and density > peak_density:
            if x > 0.1:
                break
            peak_density = density
            maxima = x

        if not peak_over and density < 0.6 * peak_density:
            peak_over = True
            peak_density = density
            min_density = density
            minima = x

        if peak_over and density > 1.5 * min_density:
            break

        if peak_over and density < min_density:
            min_density = density
            minima = x
            if density < peak_valley_ratio * peak_density:
                early_minima = x
                success = True

        x += delta_x

    if early_minima is not None and early_minima > 0.2 + peak_valley_ratio:
        early_minima = None
        success = False
        
    if early_minima is None and peak_valley_ratio > 0.55:
        early_minima = _DEFAULT_RADIUS
        success = None
        
    return success, maxima, early_minima
# end code from vamb

def find_cluster_center(matrix, cluster, medoid, avg):
    
    for sample_point in cluster:
        sample_cluster, sample_distances, average_distance = sample_medoid(matrix, sample_point, _MEDOID_RADIUS)
        if (average_distance < avg):
            #cluster = sample_cluster
            medoid = sample_point
            avg = average_distance
            
    return medoid

def get_cluster_center_improve(matrix, medoid):
    
    cluster, distances, average_distance = sample_medoid(matrix, medoid, _MEDOID_RADIUS)
    
    search_medoid = find_cluster_center(matrix, cluster, medoid, average_distance)

    while medoid != search_medoid:
        
        sampling = sample_medoid(matrix, search_medoid, _MEDOID_RADIUS)
        cluster, distances, avg = sampling
        medoid = search_medoid
        
        search_medoid = find_cluster_center(matrix, cluster, medoid, avg)
        

    return medoid, distances

def cluster_points(latent, windowsize = 200):
    matrix = normalize(latent)
    clusters = defaultdict(list)
    attempts = deque(maxlen=windowsize)
    
    peak_valley_ratio = 0.1
    successes = 0
    contig_ids = np.arange(len(matrix))
    contig_ids_ref = np.arange(len(matrix))
    
    while len(matrix) > 0:

        threshold = None
        peak = None
        medoid = None
        distances = None
        
        while threshold is None:
            seed = random.choice(contig_ids)
            medoid, distances = get_cluster_center_improve(matrix, seed)
            #distances = calc_distances(matrix, seed)
            histogram = torch.histc(distances, math.ceil(_XMAX/_DELTA_X), 0, _XMAX)
            histogram[0] -= 1
            success, peak, threshold = find_valley_ratio(histogram, peak_valley_ratio)
            
            if success is not None:
                if len(attempts) == attempts.maxlen:
                    successes -= attempts.popleft()

                successes += success
                attempts.append(success)

                if len(attempts) == attempts.maxlen and successes < MINSUCCESSES:
                    peak_valley_ratio += 0.1
                    attempts.clear()
                    successes = 0
                    
        cluster_pts = smaller_indices(distances, threshold)
        removables = smaller_indices(distances, threshold)
        removables_idx = None
        if len(removables) == 1:
            removables_idx = set([contig_ids_ref[removables]])
        else:
            removables_idx = set(contig_ids_ref[removables])
            
        if len(cluster_pts) == 1:
            clusters[contig_ids_ref[medoid]] = set([contig_ids_ref[cluster_pts]])
        else:
            clusters[contig_ids_ref[medoid]] = set(contig_ids_ref[cluster_pts])
        
        new_contig_ids_ref = np.array(
            [y for y in contig_ids_ref if y not in removables_idx])
        new_matrix = np.delete(matrix.numpy(), removables, axis=0)
        new_contig_ids = np.arange(len(new_contig_ids_ref))
        matrix = torch.from_numpy(new_matrix)
        contig_ids = new_contig_ids
        contig_ids_ref = new_contig_ids_ref
        
    return clusters


def normal(val, mean, std):
    a = np.sqrt(2*np.pi) * std
    b = -0.5 * np.square((val-mean)/std)
    b = np.exp(b)
    c = b/a + 0.0000001
    pdf = np.sum(np.log(c))

    return pdf

def filterclusters(clusters, lengthof, contig_idx_id):
    filtered_bins = dict()
    cluster_contig_id = []
    for medoid_id, contigs_id in clusters.items():
        binsize = sum(lengthof[contig_idx_id[int(contig_id)]] for contig_id in contigs_id)
        if binsize >= 200000:
            filtered_bins[medoid_id] = contigs_id
            for contig_id in contigs_id:
                cluster_contig_id.append(contig_id)

    return filtered_bins, cluster_contig_id

def perform_binning(output, contigs):
    latent = np.loadtxt(f'{output}/DRBin_latent.txt', dtype=np.float32)
    clusters = cluster_points(latent)
    clusters_output = {}
    
    contig_length = {}
    contig_id_idx = {}
    contig_idx_id = {}
    for record in SeqIO.parse(contigs, "fasta"):
        contig_length[record.id] = len(record.seq)
        contig_idx_id[len(contig_id_idx)] = record.id
        contig_id_idx[record.id] = len(contig_id_idx)
    
    filtered_bins, cluster_contig_id = filterclusters(clusters, contig_length, contig_idx_id)
    cluster_profiles = {}

    comp_profiles = np.loadtxt(f"{output}/tnfs.txt", dtype=np.float32)
    cov_profiles = np.load(f"{output}/abundance.npz")
    cov_profiles = cov_profiles['arr_0']

    for k, rs in clusters_output.items():
        vecs = []
        for r in rs:
            classified_reads.append(r)
            vecs.append(np.concatenate(
                [comp_profiles[r], cov_profiles[r]], axis=0))
        vecs = np.array(vecs)
        cluster_profiles[k] = {
            'mean': vecs.mean(axis=0),
            'std': vecs.std(axis=0),
        }
        
    classified_contigs = set(cluster_contig_id)
    all_contigs = set(range(len(comp_profiles)))
    unclassified_contigs = all_contigs - classified_contigs
 
    for r in unclassified_contigs:
        max_p = float('-inf')
        best_c = None

        for k, v in cluster_profiles.items():
            p = normal(np.concatenate(
                [comp_profiles[r], cov_profiles[r]], axis=0), v['mean'], v['std'])

            if p > max_p:
                max_p = p
                best_c = k

        if best_c is not None:
            filtered_bins[best_c].append(r)
    
    return filtered_bins