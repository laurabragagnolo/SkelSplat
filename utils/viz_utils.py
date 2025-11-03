import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
import os


def show_joints_htmp(htmp):

    # Visualize heatmaps for all joints
    fig, axes = plt.subplots(3, 6, figsize=(10, 6))
    axes = axes.ravel()
    num_joints = 17
    
    for i in range(num_joints):
        axes[i].imshow(htmp[i, ...], cmap='viridis', interpolation='bilinear')
        axes[i].set_title(f"Joint {i+1}")
        axes[i].axis('off')

    # # Plot the 2D skeleton on the first heatmap
    # ax = axes[0]
    # ax.imshow(htmp[0, ...], cmap='afmhot', interpolation='bilinear')

    plt.tight_layout()
    plt.show()

def show_single_htmp(htmp):
    # Visualize a single heatmap for a specific joint
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(htmp, cmap='viridis', interpolation='bilinear')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_rendering(render, gt_image):
    render = np.squeeze(render.detach().cpu())   
    gt_image = np.squeeze(gt_image.detach().cpu())

    # render = np.stack([render, render, render], axis=-1)
    # gt_image = np.stack([gt_image, gt_image, gt_image], axis=-1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(render)
    ax[0].axis('off')
    ax[0].set_title('Render')
    
    ax[1].imshow(gt_image)
    ax[1].axis('off')
    ax[1].set_title('Image')
    
    plt.tight_layout()
    plt.show()
    


def save_rendering(render, gt_image, out_dir, image_name, iteration):
    render = np.squeeze(render.detach().cpu())   
    gt_image = np.squeeze(gt_image.detach().cpu())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(render)
    ax[0].axis('off')
    ax[0].set_title('Render')
    
    ax[1].imshow(gt_image)
    ax[1].axis('off')
    ax[1].set_title('Image')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, image_name)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f'{out_path}/rendering_{iteration}.png')
    plt.close()


def plot_gaussians(xyz, covariance, opacity):
    xyz = xyz.reshape(4, 17, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, xyz.shape[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(xyz.shape[0]):
        for j in range(xyz.shape[1]):
            point = xyz[i, j, ...]
            ax.scatter(point[0], point[1], point[2], color=colors[i], marker='o')
            # ax.plot_wireframe(*covariance, color='b', alpha=opacity)

    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])
    plt.show()


from mpl_toolkits.mplot3d import Axes3D


def plot_3d_pose(gt_pose, pred_pose, id=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot ground truth pose in blue
    for i in range(gt_pose.shape[0]):
        point = gt_pose[i, ...]
        if i == 0:
            ax.scatter(point[0], point[1], point[2], color='green', marker='o', label='root')
        else:
            ax.scatter(point[0], point[1], point[2], color='green', marker='o')

    if pred_pose is not None:
        # Plot predicted pose in red
        for i in range(pred_pose.shape[0]):
            point = pred_pose[i, ...]
            ax.scatter(point[0], point[1], point[2], color='red', marker='o')

    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])
    plt.show()


H36M_BONES = [
    (0, 1), (1, 2), (2, 3),           # Right leg
    (0, 4), (4, 5), (5, 6),           # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
    (8, 11), (11, 12), (12, 13),      # Left arm
    (8, 14), (14, 15), (15, 16)       # Right arm
]

def plot_3d_pose_2(gt_pose, pred_pose=None, id=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot GT joints
    for i, point in enumerate(gt_pose):
        color = 'lightgreen'
        if i == 0:
            ax.scatter(*point, color=color, marker='o', label='root')
        else:
            ax.scatter(*point, color=color, marker='o')

    # Plot GT limbs
    for start, end in H36M_BONES:
        ax.plot(
            [gt_pose[start, 0], gt_pose[end, 0]],
            [gt_pose[start, 1], gt_pose[end, 1]],
            [gt_pose[start, 2], gt_pose[end, 2]],
            color='green'
        )

    if pred_pose is not None:
        # Plot predicted joints
        for point in pred_pose:
            ax.scatter(*point, color='blue', marker='o')

        # Plot predicted limbs
        for start, end in H36M_BONES:
            ax.plot(
                [pred_pose[start, 0], pred_pose[end, 0]],
                [pred_pose[start, 1], pred_pose[end, 1]],
                [pred_pose[start, 2], pred_pose[end, 2]],
                color='blue'
            )

    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([0, 1000])

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()


COCO19_BONES = [
    (0, 1),    # Neck - Nose
    (0, 3),    # Neck - lShoulder
    (3, 4),    # lShoulder - lElbow
    (4, 5),    # lElbow - lWrist
    (0, 9),    # Neck - rShoulder
    (9, 10),   # rShoulder - rElbow
    (10, 11),  # rElbow - rWrist
    (2, 6),    # BodyCenter - lHip
    (6, 7),    # lHip - lKnee
    (7, 8),    # lKnee - lAnkle
    (2, 12),   # BodyCenter - rHip
    (12, 13),  # rHip - rKnee
    (13, 14),  # rKnee - rAnkle
    (1, 15),   # Nose - rEye
    (15, 17),  # rEye - rEar
    (1, 16),   # Nose - lEye
    (16, 18),  # lEye - lEar
    (2, 0),    # BodyCenter - Neck
]

def plot_3d_pose_3(gt_pose, pred_pose=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def swap_axes(pose):
        # Convert from (x, y, z) to (x, z, y) to make y-axis point up
        return pose[:, [0, 2, 1]]

    def ground_pose(pose):
        # Shift pose so minimum y (vertical axis) is zero (ground level)
        min_y = np.min(pose[:, 1])
        pose[:, 1] -= min_y
        return pose

    def plot_joints(pose, color):
        for point in pose:
            ax.scatter(*point, color=color, marker='o')

    def plot_bones(pose, color):
        for start, end in COCO19_BONES:
            if start < len(pose) and end < len(pose):
                ax.plot(
                    [pose[start, 0], pose[end, 0]],
                    [pose[start, 1], pose[end, 1]],
                    [pose[start, 2], pose[end, 2]],
                    color=color
                )

    # Ground truth
    if gt_pose is not None:
        if gt_pose.shape[0] <= 18:
            pelvis = (gt_pose[8] + gt_pose[11]) / 2
            gt_pose = np.vstack([gt_pose, pelvis])
        gt_pose = swap_axes(gt_pose)
        gt_pose = ground_pose(gt_pose)
        plot_joints(gt_pose, color='lightgreen')
        plot_bones(gt_pose, color='green')

    # Prediction
    if pred_pose is not None:
        if pred_pose.shape[0] <= 18:
            pelvis = (pred_pose[8] + pred_pose[11]) / 2
            pred_pose = np.vstack([pred_pose, pelvis])
        pred_pose = swap_axes(pred_pose)
        pred_pose = ground_pose(pred_pose)
        plot_joints(pred_pose, color='blue')
        plot_bones(pred_pose, color='blue')

    ax.set_xlim([-1500, 1500])
    ax.set_ylim([-1500, 1500])
    ax.set_zlim([-1500, 200])  # Z is now the vertical axis

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()


def plot_2d_pose(gt_pose, pred_pose, id=0):
    fig, ax = plt.subplots()
    # Plot ground truth pose in blue
    for i in range(gt_pose.shape[0]):
        point = gt_pose[i, ...]
        if i == 0:
            ax.scatter(point[0], point[1], color='blue', marker='o', label='root')
        else:
            ax.scatter(point[0], point[1], color='blue', marker='o')
    
    # Plot predicted pose in red
    for i in range(pred_pose.shape[0]):
        point = pred_pose[i, ...]
        ax.scatter(point[0], point[1], color='red', marker='o')
    
    ax.set_aspect('equal', 'box')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.show()


def plot_3d_gaussians(means, scaling, opacity, color='blue', n_std=2):

    means = means.detach().cpu().numpy()   
    scaling = scaling.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()

    print(scaling.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop over each Gaussian mean
    for mean in means:
        # Create a diagonal covariance matrix from the variances
        cov = np.diag(scaling)  # Variances are provided directly

        # Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = eigh(cov)
        # Sort eigenvalues and eigenvectors in descending order
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # Generate data for a unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Transform the sphere to match the Gaussian ellipsoid
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Scale and rotate the point, then shift by the mean
                point = n_std * np.sqrt(eigvals) * np.array([x[i, j], y[i, j], z[i, j]])
                transformed_point = eigvecs @ point + mean
                x[i, j], y[i, j], z[i, j] = transformed_point

        # Plot the ellipsoid surface
        ax.plot_surface(x, y, z, color=color, alpha=opacity, rstride=4, cstride=4, linewidth=0)

    # Set axis labels for better visualization
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
