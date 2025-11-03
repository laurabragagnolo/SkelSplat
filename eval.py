
import os
import numpy as np
import open3d as o3d
import hydra
from omegaconf import DictConfig
from arguments.config_handler import ConfigHandler, TriangulationConfigHandler


def align_pred_cpn(pred_coords, gt_coords, image_relpaths):
    # Align the predicted 3D poses with the ground truth
    start_poses = 0
    count = 0
    for i, path in enumerate(image_relpaths):
        if 'S11' in path and 'Directions.' in path:
            start_poses = i
            count += 1
    insert_poses = np.zeros((count, 17, 3))
    new_pred_coords = np.vstack((pred_coords[:start_poses], insert_poses, pred_coords[start_poses:]))
    return new_pred_coords

def get_pred_coords_h36m(ply_dir, sorted_entries, absolute=False, cpn=False):

    activities = []
    pred_coords = []
    for entry in sorted_entries:
        subject, activity, frame = entry
        if absolute:
            if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                continue
        ply_file = f'{ply_dir}/{subject}_{activity}_{frame}'
        pcd = o3d.io.read_point_cloud(ply_file)
        pred_coords.append(np.asarray(pcd.points))
        activities.append(activity.split(" ")[0])

    pred_coords = np.array(pred_coords)
    activities = np.array(activities)
    # print(pred_coords.shape)
    return pred_coords, activities

def get_pred_coords(ply_dir, sorted_entries, absolute=False):

    pred_coords = []
    for entry in sorted_entries:
        subject, activity, frame = entry
        ply_file = f'{ply_dir}/{subject}_{activity}_{frame}'
        pcd = o3d.io.read_point_cloud(ply_file)
        pred_coords.append(np.asarray(pcd.points))

    pred_coords = np.array(pred_coords)
    # print(pred_coords.shape)
    return pred_coords


def get_gt_poses_h36m(gt_path, absolute=False, cpn=False, frame_step=64):
    gt_poses = []
    for subject in sorted(os.listdir(gt_path)):
        if not subject.startswith('S'):
            continue
        for activity in sorted(os.listdir(f'{gt_path}/{subject}')):
            if absolute:
                if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                    continue
            if cpn:
                if subject == 'S11' and activity == 'Directions':
                    continue
            gt_3d = np.load(f'{gt_path}/{subject}/{activity}/poses.npz')['poses']
            gt_sampled = gt_3d[::frame_step]
            gt_poses.append(gt_sampled)
    gt_poses = np.concatenate(gt_poses, axis=0)
    return gt_poses


def get_gt_poses(gt_path, absolute=False, dataset="panoptic", frame_step=1, nviews=4):
    
    gt_poses = []
    for subject in sorted(os.listdir(gt_path)):
        if not subject.startswith('S'):
            continue
        for activity in sorted(os.listdir(f'{gt_path}/{subject}')):
            if dataset == "panoptic":
                gt_3d = np.load(f'{gt_path}/{subject}/{activity}/poses_filtered_{nviews}.npz', allow_pickle=True)['poses']
            else:
                gt_3d = np.load(f'{gt_path}/{subject}/{activity}/poses.npz', allow_pickle=True)['poses3d']
            gt_sampled = gt_3d[::frame_step]
            gt_poses.append(gt_sampled)

    gt_poses = np.concatenate(gt_poses, axis=0)
    return gt_poses

def evaluate(gt_path, output_path, iterations, start_id, end_id, cpn=False):

    for it in iterations:
        print(f"Results for {it} iterations \n")
        # load the predicted 3D poses
        ply_dir = f'{output_path}/point_cloud/iteration_{it}'
        entries = os.listdir(ply_dir)
        name_parts = [entry.split('_') for entry in entries]
        if "panoptic" in gt_path:
            name_parts = [[entry.split("_")[0], entry.split("_")[1] + "_" + entry.split("_")[2], entry.split("_")[-1]] for entry in entries]
            dataset = "panoptic"
        if "occlusion-person" in gt_path:
            name_parts = [[entry.split("_")[0], entry.split("_")[1], entry.split("_")[-1]] for entry in entries]
            dataset = "occlusion-person"

        sorted_entries = sorted(name_parts)


        if "h36m" in gt_path:

            ordered_activities = (
                'Directions Discussion Eating Greeting Phoning Posing Purchases ' +
                'Sitting SittingDown Smoking Photo Waiting Walking WalkDog WalkTogether').split()

            # Absolute MPJPE
            absolute = True
            gt_coords = get_gt_poses_h36m(gt_path, absolute, cpn, frame_step=64)
            pred_coords, activities = get_pred_coords_h36m(ply_dir, sorted_entries, absolute, cpn)
            if end_id > pred_coords.shape[0]:
                end_id = pred_coords.shape[0]

            print("Evaluating scenes from", start_id, "to", end_id)
            abs_error = np.linalg.norm(gt_coords[start_id:end_id, ...] - pred_coords[start_id:end_id, ...], axis=-1)
            abs_error_mean = np.mean(abs_error)
            print("Absolute MPJPE: ", np.round(abs_error_mean, 2))
            activities_errors = [np.mean(abs_error[a == activities]) for a in ordered_activities]
            print(np.round(activities_errors, 2))

            # Relative MPJPE
            absolute = False
            gt_coords = get_gt_poses_h36m(gt_path, absolute, cpn, frame_step=64)
            pred_coords, activities = get_pred_coords_h36m(ply_dir, sorted_entries, absolute, cpn)
            if end_id < pred_coords.shape[0]:
                end_id = pred_coords.shape[0]
            # align the root joint
            gt_coords -= gt_coords[:, 0, np.newaxis]
            pred_coords -= pred_coords[:, 0, np.newaxis]
            rel_error = np.linalg.norm(gt_coords[start_id:end_id, ...] - pred_coords[start_id:end_id, ...], axis=-1)
            rel_error_mean = np.mean(rel_error)
            print("Relative MPJPE: ", np.round(rel_error_mean, 2))
            activities_errors = [np.mean(rel_error[a == activities]) for a in ordered_activities]
            print(np.round(activities_errors, 2))
            print("\n")

        else:

            # Absolute MPJPE
            absolute = True
            gt_coords = get_gt_poses(gt_path, absolute, dataset, frame_step=1)
            pred_coords = get_pred_coords(ply_dir, sorted_entries, absolute)
            if end_id > pred_coords.shape[0]:
                end_id = pred_coords.shape[0]

            print("Evaluating scenes from", start_id, "to", end_id)
            abs_error = np.linalg.norm(gt_coords[start_id:end_id, ...] - pred_coords[start_id:end_id, ...], axis=-1)
            abs_error_mean = np.mean(abs_error)
            print("Absolute MPJPE: ", np.round(abs_error_mean, 2))

            # Relative MPJPE
            absolute = False
            gt_coords = get_gt_poses(gt_path, absolute, dataset, frame_step=1)
            pred_coords = get_pred_coords(ply_dir, sorted_entries, absolute)
            if end_id < pred_coords.shape[0]:
                end_id = pred_coords.shape[0]
            # align the root joint
            gt_coords -= gt_coords[:, 0, np.newaxis]
            pred_coords -= pred_coords[:, 0, np.newaxis]
            rel_error = np.linalg.norm(gt_coords[start_id:end_id, ...] - pred_coords[start_id:end_id, ...], axis=-1)
            rel_error_mean = np.mean(rel_error)
            print("Relative MPJPE: ", np.round(rel_error_mean, 2))
            print("\n")



@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(cfg: DictConfig):

    if "training" in cfg: 
        config = ConfigHandler(cfg)
    else:
        config = TriangulationConfigHandler(cfg)

    output_path = config.hydra_out
    dataset = cfg.dataset
    start_id = dataset.start_scene_id
    end_id = dataset.end_scene_id
    debug = cfg.debug

    print("Evaluating ", output_path)

    gt_path = os.path.join(dataset.data_root, "3d_gt")
    iterations = debug.save_iterations
    evaluate(gt_path, output_path, iterations, start_id, end_id, dataset.poses_2d == "cpn")


if __name__ == "__main__":
    main()

