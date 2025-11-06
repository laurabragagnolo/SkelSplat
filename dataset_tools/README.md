# ⚙️ Data Preprocessing

This folder contains dataset-specific preprocessing utilities used to prepare inputs for training and evaluation.

It standardizes:

* **Directory layout** and symlinks per dataset
* **Ground-truth parsing** (subjects, actions, camera params)
* **2D keypoints** aggregation per view
* **Initial 3D pose guesses** (from monocular 3D or triangulation)

## Human3.6M Dataset

**1. Preliminaries**

Download & structure Human3.6M using [h36m-fetch](https://github.com/anibali/h36m-fetch/tree/master). Only test subjects (S9 and S11) required.

Assumed layout (typical from h36m-fetch):

```
/path/to/h36m/
├── S9/
    ├── Images
    ├── MyPoseFeatures
        ├── D2_Positions
        ├── D3_Positions
    ....
├── S11/
    ├── Images
    ├── MyPoseFeatures
    ....
└── Release-v1.2/...
```

**2. Organize Ground Truth**

Parse subjects/actions, camera intrinsics/extrinsics, and 3D joints into the unified schema.

```bash
cd h36m
python preprocess_h36m_gt.py --root_dir /path/to/h36m --output_dir "../../data/h36m"
```
This creates:
```
data
├── h36m/
    ├── 2g_gt/
        ├── S9/
            ├── Directions/
                ├── 54138969/
                    ├── poses.npz
                ├── 55011271/
                ├── 58860488/
                ├── 60457274/
            ├── Directions 1/
            ...
        ├── S11/
            ├── Directions/
            ├── Directions 1/
            ...
    ├── 3g_gt/
        ├── cameras/
        ├── S9/
            ├── Directions/
                ├── poses.npz
            ├── Directions 1/
                ├── poses.npz
            ...
        ├── S11/
            ...
```

**3. Get 2D Keypoints per View**

Obtain 2D detections for every camera and frame (e.g., ResNet as in [AdaFuse](https://github.com/zhezh/adafuse-3d-human-pose), CPN as in [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md), or [Metrabs](https://github.com/isarandi/metrabs)). Note that Metrabs will generate predictions for both 2D and 3D poses per view and frame.
Organize such data using the provided scripts into the same format used for GT.
```bash
python preprocess_resnet_2d_poses.py
python preprocess_cpn_2d_poses.py
python preprocess_metrabs_predictions.py
```

You are free to use your own detector; only the on-disk format matters.

**4. Generate 3D Initial Guess**

Create initial guess 3D poses using either monocular 3D pose fusion or simple triangulation of extracted 2D poses
For triangulation:
```bash
cd ..
cd ..
python triangulation.py --config-name triangulation.yaml
cd dataset_tools
python preprocess_triang_initial_guess.py 
```

Alternatively, you can use as initial guess the fusion of 3D monocular detections given by Metrabs, following [this work](https://link.springer.com/chapter/10.1007/978-3-031-92591-7_8).
You can do this by simply using this script:
```bash
cd h36m
python compute_initial_guess.py
```

## Human3.6M-Occ Dataset

Generate occluded versions (Occ-2 / Occ-3 / Occ-3-Hard) of Human3.6M following [this repo](https://github.com/laurabragagnolo/human3.6m-occluded).
Organize data and produce initial guess following the procedure described for Human3.6M.

## CMU Panoptic Dataset

**1. Preliminaries**

Download [CMU Panoptic Dataset](http://domedb.perception.cs.cmu.edu/).
We use only single-person sequences, following [this work](http://domedb.perception.cs.cmu.edu/mtc.html). Test sequences include: 171204_pose5, 171204_pose6.
We select a total of 8 cameras (more details in the paper): "00_01", "00_02", "00_10", "00_13", "00_03", "00_23", "00_19", "00_30".

Assumed layout:

```
/path/to/panoptic/
├── 171204_pose5/
    ├── hdImages/
        ├── 00_01/
        ├── 00_02/
        ├── 00_03/
        ├── 00_10/
    ...
    ├── hdPose3d_stage1_coco19/
        ├── body3DScene_00000000.json
        ....
├── 171204_pose6/
    ...
```

**2. Organize Ground Truth**

Parse 3D ground truth and create 2D ground truth via reprojection on each view:

```bash
cd panoptic
python preprocess_panoptic_gt.py
```

We take "S0" as fake subject to keep a structure similar to the one obtained for the other datasets.

**3. Get 2D Keypoints per View**

Obtain 2D detections for every camera and frame using [Metrabs](https://github.com/isarandi/metrabs).
Organize such data using the provided scripts.  
```bash
python preprocess_metrabs_predictions.py
python filter_preds_number_views.py    # to ensure that each view is associated to a detection (not None)
```

**4. Generate 3D Initial Guess**

Create initial guess 3D poses using either monocular 3D pose fusion or simple triangulation of extracted 2D poses, as described for Human3.6M.

## Occlusion-Person Dataset

**1. Preliminaries**

Download [Occlusion-Person](https://github.com/zhezh/occlusion_person).

**2. Organize Ground Truth**

Parse ground truth and obtain consistent structure as done for other datasets.

```bash
cd occlusion-person
python preprocess_occlusion_person_gt.py
```

We take "S0" as fake subject to keep a structure similar to the one obtained for the other datasets.

**3. Get 2D Keypoints per View**

Obtain 2D detections for every camera and frame using a ResNet, as done in [AdaFuse](https://github.com/zhezh/adafuse-3d-human-pose).
Organize such data using the provided scripts.  
```bash
python preprocess_resnet_2d_poses.py
```

**4. Generate 3D Initial Guess**

Create initial guess via simple triangulation of extracted 2D poses, as described for Human3.6M.

## 🔎 Check your data

To check and visualize your preprocessed 2D and 3D data, use the scripts available in the `dataset_tools/` directory.

```bash
python check_2d_dataset.py --gt_dir /your/path/to/gt_2d --pred_dir /your/path/to/pred_2d

python check_3d_dataset.py --gt_dir /your/path/to/gt_3d --initial_guess_dir /your/path/to/guess_3d
```
