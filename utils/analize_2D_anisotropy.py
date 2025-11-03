import json
import numpy as np
import matplotlib.pyplot as plt



def percent_inside_sigmas(means, covs, gt, ks=(1,2,3)):
    """
    means: (N,3) array of Gaussian means
    covs:  (N,3,3) array of covariance matrices
    gt:    (N,3) array of ground-truth points
    ks:    iterable of sigma-levels to test (default 1,2,3)
    Returns: dict[k] = fraction of points with Mahalanobis distance <= k
    """
    N = means.shape[0]
    # Precompute inverse covariances
    inv_covs = np.linalg.inv(covs)  # shape (N,3,3)

    # Compute Mahalanobis distances
    deltas = gt - means         # shape (N,3)
    # for each i, d2_i = deltas[i].T @ inv_covs[i] @ deltas[i]
    # vectorize with einsum:
    d2 = np.einsum('ni,nij,nj->n', deltas, inv_covs, deltas)  # shape (N,)

    results = {}
    for k in ks:
        inside = np.sum(d2 <= k**2)
        results[k] = inside / N
    return results





def compute_confidence_per_joint(file_path):
    
    with open("lambdas.json", "r") as f:
        data = json.load(f)


    for scene in data:
        lambdas = scene["lambdas"]

        anisotropies = {}  # joint_id -> list of anisotropy values (one per view)

        for joint_id, view_lambdas in lambdas.items():
            anisotropies[joint_id] = []
            for lambdas_view in view_lambdas:
                λ1, λ2 = lambdas_view
                anisotropy = max(λ1, λ2) / min(λ1, λ2)
                anisotropies[joint_id].append(anisotropy)

        print(f"Scene {scene['scene_id']} Anisotropies:{anisotropies}")

    
    
    # plt.figure(figsize=(12, 6))
    
    # # Plot error vs confidence
    # plt.subplot(1, 2, 1)
    # plt.scatter(traces, j_errors, alpha=0.5)
    # plt.title('Error vs Trace')
    # plt.xlabel('Trace')
    # plt.ylabel('Error')
    
    # # Plot error vs anisotropy
    # plt.subplot(1, 2, 2)
    # plt.scatter(anisotropies, j_errors, alpha=0.5)
    # plt.title('Error vs Anisotropy')
    # plt.xlabel('Anisotropy')
    # plt.ylabel('Error')

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    file_path = 'lambdas.json'
    confidence_per_joint = compute_confidence_per_joint(file_path)
   