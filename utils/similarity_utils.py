import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations




def pairwise_cosine_similarity(gradients):
    V, N, D = gradients.shape
    pairwise_similarity = torch.zeros((N, V, V), device=gradients.device)
    
    for i in range(N):
        grads = gradients[:, i, :]  # (V, D)
        norms = torch.norm(grads, dim=1, keepdim=True) + 1e-8  # Normalize gradients
        normalized_grads = grads / norms
        
        # Compute pairwise cosine similarity
        for v1, v2 in combinations(range(V), 2):
            cos_sim = torch.dot(normalized_grads[v1], normalized_grads[v2])
            pairwise_similarity[i, v1, v2] = cos_sim
            pairwise_similarity[i, v2, v1] = cos_sim
        
        # Diagonal should be 1 (each view compared to itself)
        pairwise_similarity[i].fill_diagonal_(1.0)
    
    return pairwise_similarity


def pairwise_cosine_norm_similarity(gradients, w1=0.2, w2=0.8):
    V, N, D = gradients.shape
    pairwise_similarity = torch.zeros((N, V, V), dtype=torch.float32, device=gradients.device)
    
    for i in range(N):
        grads = gradients[:, i, :]  # Shape (V, D), gradients for joint i across views
        
        # Compute normalization factor: sum of norms across all views
        norm_factors = torch.norm(grads, p=2, dim=1, keepdim=True)  # Shape (V, 1)
        total_norm = norm_factors.sum()  # Scalar: sum of norms over all views
        
        # Normalize gradients
        if total_norm > 0:
            grads = grads / total_norm  # Normalize each vector by total sum of norms
        
        for v1, v2 in combinations(range(V), 2):
            g1, g2 = grads[v1], grads[v2]  # Get normalized D-dimensional vectors for views v1 and v2
            
            # Compute cosine similarity
            cosine_sim = torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2) + 1e-8)
            
            # Compute relative difference in norm
            norm_g1 = torch.norm(g1)
            norm_g2 = torch.norm(g2)
            relative_diff = torch.abs(norm_g1 - norm_g2) / (torch.max(norm_g1, norm_g2) + 1e-8)
            
            # Compute weighted similarity score (w1 * cosine similarity - w2 * relative norm difference)
            similarity_score = w1 * cosine_sim - w2 * relative_diff
            
            pairwise_similarity[i, v1, v2] = similarity_score
            pairwise_similarity[i, v2, v1] = similarity_score  # Ensure symmetry
        
        pairwise_similarity[i].fill_diagonal_(1.0)
    
    return pairwise_similarity  # Shape (N, V, V)



def identify_consistent_views(pairwise_similarity, threshold=0.5):
    N, V, _ = pairwise_similarity.shape
    consistent_views = torch.zeros((N, V), dtype=torch.bool, device=pairwise_similarity.device)
    
    for i in range(N):
        sim_matrix = pairwise_similarity[i] >= threshold  # Thresholded similarity
        view_counts = sim_matrix.sum(dim=1) - 1  # Count how many views each view agrees with
        
        # Select views with at least 2 agreements
        consistent_views[i] = view_counts >= 2
    
    return consistent_views


# import seaborn as sns

# def plot_pairwise_similarity(pairwise_similarity, point_idx=0):
#     """
#     Plot a heatmap of the pairwise similarity matrix for a given 3D point.

#     :param pairwise_similarity: (N, 4, 4) cosine similarity matrix for all views.
#     :param point_idx: Index of the 3D point to visualize.
#     """
#     plt.figure(figsize=(6, 5))
#     ax = sns.heatmap(pairwise_similarity[point_idx], annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")

#     plt.title(f"Pairwise Gradient Similarity (3D Point {point_idx + 1})")
#     plt.xlabel("View Index")
#     plt.ylabel("View Index")
#     plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=["V1", "V2", "V3", "V4"])
#     plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=["V1", "V2", "V3", "V4"], rotation=0)


# from mpl_toolkits.mplot3d import Axes3D

# def plot_gradient_vectors(gradients, consistent_views, point_idx=0):
#     """
#     Plot the 3D gradient vectors from different views for a given 3D point.
    
#     :param gradients: (N, 4, 3) gradient vectors for N 3D points across 4 views.
#     :param consistent_views: (N, 4) boolean matrix indicating inlier views (True) and outliers (False).
#     :param point_idx: Index of the 3D point to visualize.
#     """
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Get gradient vectors for the selected 3D point
#     vectors = gradients[:, point_idx, :]

#     inlier_vectors = vectors[consistent_views[..., point_idx].cpu().numpy()]  # Filter inlier vectors

#     mean_vector = np.mean(inlier_vectors, axis=0)  # Mean of all vectors
#     sum_vector = np.sum(inlier_vectors, axis=0)    # Sum of all vectors
    
#     # Assign a unique color per view
#     view_colors = ["blue", "green", "orange", "purple"]
#     labels = ["V1", "V2", "V3", "V4"]

#     # Plot each gradient vector
#     for v in range(4):
#         is_inlier = consistent_views[v, point_idx]
#         alpha = 1.0 if is_inlier else 0.3  # Reduce opacity for outliers
#         linestyle = "-" if is_inlier else "--"  # Dashed lines for outliers

#         ax.quiver(0, 0, 0, vectors[v, 0], vectors[v, 1], vectors[v, 2], 
#                   color=view_colors[v], label=f"{labels[v]} ({'Inlier' if is_inlier else 'Outlier'})",
#                   linewidth=2, alpha=alpha)
        
#         # Plot the mean vector
#         ax.quiver(0, 0, 0, mean_vector[0], mean_vector[1], mean_vector[2], 
#                 color="red", label="Mean Vector", linewidth=3, linestyle="-")

#         # Plot the sum vector
#         ax.quiver(0, 0, 0, sum_vector[0], sum_vector[1], sum_vector[2], 
#                 color="cyan", label="Sum Vector", linewidth=3, linestyle="-")

#     # Set labels and title
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title(f"Gradient Vectors for 3D Point {point_idx + 1}")

#     # Set limits
#     max_range = np.max(np.abs(vectors)) * 1.2
#     ax.set_xlim(-max_range, max_range)
#     ax.set_ylim(-max_range, max_range)
#     ax.set_zlim(-max_range, max_range)
    
#     ax.legend()


def compute_scaling_weights(similarity_matrix):

    self_similarities = torch.diagonal(similarity_matrix, dim1=1, dim2=2)
    similarities = (similarity_matrix.sum(dim=2) - self_similarities) / 3

    # scaling_weights = torch.log(similarities + 2) / torch.log(torch.tensor(3.0, device="cuda"))
    scaling_weights = weight_function(similarities)
    scaling_weights = torch.transpose(scaling_weights, 0, 1)

    return scaling_weights


def weight_function(s):
    """
    Computes the piecewise function:
      - Linear for -1 <= x < 0: y = 0.8 * (x + 1)
      - Logarithmic for 0 <= x <= 1: y = 0.54 * log_3(x+2) + 0.46
    """
    weights = torch.zeros_like(s)  # Initialize output tensor
    
    # Logarithmic part for x in [0,1]
    mask_log = (s >= 0) & (s <= 1)
    weights[mask_log] = 0.54 * (torch.log(s[mask_log] + 2) / torch.log(torch.tensor(3.0))) + 0.46
    
    # Linear part for x in [-1,0)
    mask_lin = (s >= -1) & (s < 0)
    weights[mask_lin] = 0.8 * (s[mask_lin] + 1)
    
    return weights


def select_views(error_matrix, threshold=2.5, min_views=4):
    
    selected_views = error_matrix <= threshold
    
    for joint_idx in range(error_matrix.shape[1]):
        if torch.sum(selected_views[:, joint_idx]) < min_views:
            sorted_values, sorted_indices = torch.sort(error_matrix[:, joint_idx], descending=False)
            selected_views[sorted_indices[:min_views], joint_idx] = True

    view_scores = torch.sum(selected_views, dim=1)  # Count trues per view
    best_views = torch.argsort(view_scores, descending=True)[:min_views]

    final_matrix = torch.zeros_like(selected_views, dtype=torch.bool)
    final_matrix[best_views, :] = True
    
    return selected_views, best_views, final_matrix