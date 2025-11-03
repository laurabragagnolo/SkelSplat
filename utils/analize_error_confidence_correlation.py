import json
import numpy as np
import matplotlib.pyplot as plt



def plot_percent_inside_sigmas(results, ks=(1, 2, 3), title="Percent of GT inside k-sigma"):
    joint_names = list(results.keys())
    num_joints = len(joint_names)

    # Prepare data per sigma
    data = {k: [results[joint][k] * 100 for joint in joint_names] for k in ks}

    # Bar width and x locations
    bar_width = 0.25
    x = np.arange(num_joints)

    # Plot each sigma group with an offset
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    for i, k in enumerate(ks):
        ax.bar(x + i * bar_width, data[k], width=bar_width, label=f'{k}σ', color=colors[i])

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=20)
    ax.set_ylabel("Percentage of GT joints", fontsize=20)
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=30)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()



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



def get_means_covs_gt(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    means = []
    covs = []
    gt = []

    for scene in data:
        for joint in scene['info']:
            dict = scene['info'][joint]
            means.append(dict['3d_pred'])
            covs.append(dict['covariance'])
            gt.append(dict['3d_gt'])

    means = np.array(means)
    covs = np.array(covs)
    gt = np.array(gt)

    return means, covs, gt


def percent_inside_sigmas_per_joint(means, covs, gt, joint_names, ks=(1,2,3)):
    """
    means: (N, J, 3)
    covs:  (N, J, 3, 3)
    gt:    (N, J, 3)
    joint_names: list of J joint names
    ks:    tuple of sigma thresholds

    Returns: dict[joint_name] = dict[k] = percentage inside k-sigma
    """
    N, J, _ = means.shape
    results = {}

    for j in range(J):
        mu_j   = means[:, j, :]     # (N, 3)
        cov_j  = covs[:, j, :, :]   # (N, 3, 3)
        gt_j   = gt[:, j, :]        # (N, 3)
        delta  = gt_j - mu_j        # (N, 3)

        inv_cov_j = np.linalg.inv(cov_j)  # (N, 3, 3)
        d2 = np.einsum('ni,nij,nj->n', delta, inv_cov_j, delta)  # (N,)

        res = {}
        for k in ks:
            res[k] = np.mean(d2 <= k**2)
        results[joint_names[j]] = res

    return results



def analyze_error_confidence_correlation(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    errors = []
    joint_errors = []
    confidences = []
    anisotropies = []
    traces = []
    eigenvalues = []
    covariances = []

    for scene in data:
        for joint in scene['info']:
            dict = scene['info'][joint]
            errors.append(dict['error'])
            joint_errors.append(dict['joint_errors'])
            anisotropies.append(dict['anisotropy'])
            traces.append(dict['trace'])
            eigenvalues.append(dict['eigenvalues'])
            covariances.append(dict['covariance'])


    errors = np.array(errors)
    traces = np.array(traces)

    determinants = np.array([np.linalg.det(cov) for cov in covariances])
    # anisotropies = np.array(anisotropies)

    j_errors = np.mean(np.array(joint_errors), axis=1)

    # if errors.shape[1] > 1:
    #     errors = np.max(errors, axis=1)    

    # print(np.max(errors))
    # print(np.min(errors))
    # print(np.mean(errors))

    # errors_selected = errors < 10.0
    # print(errors_selected[:10])
    # errors = errors[errors_selected]
    # confidences = confidences[errors_selected]

    # print(errors.shape, confidences.shape)

    plt.figure(figsize=(12, 6))
    
    # Plot error vs confidence
    plt.subplot(1, 2, 1)
    plt.scatter(traces, errors, alpha=0.5)
    plt.title('Error vs Trace')
    plt.xlabel('Trace')
    plt.ylabel('Error')
    
    # Plot error vs anisotropy
    plt.subplot(1, 2, 2)
    plt.scatter(traces, j_errors, alpha=0.5)
    plt.title('Joints Error vs Trace')
    plt.xlabel('Trace')
    plt.ylabel('Joints Error')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'info_confidences_h36m_ada_occ.json'
    means, covs, gt = get_means_covs_gt(file_path)
    covs = covs.reshape(-1, 3, 3)  # Ensure covs is (N, 3, 3)
    percent_inside_sigmas = percent_inside_sigmas(means, covs, gt)

    means_joints = means.reshape(-1, 17, 3)  # Assuming 17 joints
    covs_joints = covs.reshape(-1, 17, 3, 3)
    gt_joints = gt.reshape(-1, 17, 3)

    percent_inside_sigmas_per_joint = percent_inside_sigmas_per_joint(
        means_joints, covs_joints, gt_joints,
        joint_names=['root', 'lhip', 'lknee', 'lfoot', 'rhip', 'rknee', 'rfoot', 'spine', 'thorax', 'neck', 'head', 'rshoulder', 'relbow', 'rhand', 'lshoulder', 'lelbow', 'lhand']
    )
    plot_percent_inside_sigmas(percent_inside_sigmas_per_joint, ks=(1, 2, 3), title="Percent of GT inside k-sigma for each joint")

    print("Percent inside sigmas for all joints:" , percent_inside_sigmas_per_joint)

    print("Percent inside sigmas:", percent_inside_sigmas)

    