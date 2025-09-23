import pandas as pd
import yaml
import torch
from model import YNet
import numpy as np
from MPDC_head import Network

CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = 'data/SDD/train_trajnet.pkl'
TRAIN_IMAGE_PATH = 'data/SDD/train'

OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a
ROUNDS = 1
BATCH_SIZE = 1

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
print(params)

df_train = pd.read_pickle(TRAIN_DATA_PATH)

# Define incremental tasks
scene_splits = {
    "nexus": ["nexus_0", "nexus_1", "nexus_2", "nexus_3", "nexus_4", "nexus_7", "nexus_8", "nexus_9"],
    "coupa": ["coupa_3"],
    "gates": ["gates_0", "gates_1", "gates_3", "gates_4", "gates_5", "gates_6", "gates_7", "gates_8"],
    "hyang": ["hyang_4", "hyang_5", "hyang_6", "hyang_7", "hyang_9"]
}

scene_column_index = 4
train_scene_datasets = {}
for scene, scene_ids in scene_splits.items():
    train_scene_datasets[scene] = df_train[df_train.iloc[:, scene_column_index].isin(scene_ids)]

for scene, data in train_scene_datasets.items():
    print(f"Scene: {scene}, Data Shape: {data.shape}")

# Define the training set and validation set
group_size = 20
train_scene_datasets = {}
val_scene_datasets = {}

for scene, scene_ids in scene_splits.items():
    scene_data = df_train[df_train.iloc[:, scene_column_index].isin(scene_ids)]

    data_array = scene_data.values
    num_groups = len(data_array) // group_size
    data_array = data_array[:num_groups * group_size]
    groups = data_array.reshape((num_groups, group_size, -1))

    np.random.seed(42)
    np.random.shuffle(groups)
    train_size = int(0.8 * num_groups)
    train_groups = groups[:train_size]
    val_groups = groups[train_size:]

    train_data = train_groups.reshape(-1, data_array.shape[1])
    val_data = val_groups.reshape(-1, data_array.shape[1])

    train_scene_datasets[scene] = pd.DataFrame(train_data, columns=scene_data.columns)
    val_scene_datasets[scene] = pd.DataFrame(val_data, columns=scene_data.columns)

# For each scene, 100 random trajectories are selected from the validation set, effectively defining a batch.
sampled_val_scene_datasets = {}

for scene, val_data in val_scene_datasets.items():
    val_array = val_data.values
    num_val_groups = len(val_array) // group_size

    val_array = val_array[:num_val_groups * group_size]
    val_groups = val_array.reshape((num_val_groups, group_size, -1))

    np.random.seed(42)
    sampled_num = min(100, len(val_groups))
    sampled_indices = np.random.choice(len(val_groups), size=sampled_num, replace=False)
    sampled_groups = val_groups[sampled_indices]

    sampled_data = sampled_groups.reshape(-1, val_array.shape[1])
    sampled_val_scene_datasets[scene] = pd.DataFrame(sampled_data, columns=val_data.columns)

for scene in train_scene_datasets:
    print(
        f"Scene: {scene}, Train Shape: {train_scene_datasets[scene].shape}, Val Shape: {sampled_val_scene_datasets[scene].shape}")

val_scene_datasets = sampled_val_scene_datasets


scenes = ['nexus', 'coupa', 'gates', 'hyang']
model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
MPDC_backbone = Network(12288, 512, 128, params['class_scene'])

scenes_loop = list(train_scene_datasets.keys())
incremental_results = {}
incremental_results_before = {}
repaly_sample = None
threshold = []
prototype = []
covariance = []

# Detect each novelty introduction during traversal
for i, current_scene in enumerate(scenes_loop):

    if i == (len(scenes)-1):
        break

    print(f"-------------------Detection after Accommodating Scene: {current_scene}----------------------")

    if i > 0:
        model.model.encoder.update_lora()

        load_path_nsp = 'trained_models/' + scenes[i] + '_weights.pt'
        model.load(load_path_nsp)

        load_path_nsp = 'trained_models/' + scenes[i] + '_MPDC_weights.pt'
        checkpoint_trained = torch.load(load_path_nsp)
        MPDC_backbone.load_state_dict(checkpoint_trained)
    else:
        load_path_nsp = 'trained_models/' + scenes[i] + '_weights.pt'
        model.load(load_path_nsp)

        load_path_nsp = 'trained_models/' + scenes[i] + '_MPDC_weights.pt'
        checkpoint_trained = torch.load(load_path_nsp)
        MPDC_backbone.load_state_dict(checkpoint_trained)

    current_train_data = train_scene_datasets[current_scene]
    current_val_data = val_scene_datasets[current_scene]

    # Calculate the p-quantile threshold
    mean_score = model.OOD_update(
        current_train_data,
        repaly_sample,
        params,
        MPDC_backbone,
        train_image_path=TRAIN_IMAGE_PATH,
        batch_size=BATCH_SIZE,
        device=None,
        dataset_name=DATASET_NAME,
        task_id=i
    )
    threshold.append(mean_score)

    # Detection
    print(f"---------------Detect subsequent tasks-------------")
    ADE_results = {}
    FDE_results = {}
    save_goal = None

    MPDC_backbone_list = []
    for temp in range(i + 1):
        MPDC_backbone_temp = Network(12288, 512, 128, params['class_scene'])
        load_path_nsp = 'trained_models/' + scenes[temp] + '_MPDC_weights.pt'
        checkpoint_trained = torch.load(load_path_nsp)
        MPDC_backbone_temp.load_state_dict(checkpoint_trained)
        MPDC_backbone_list.append(MPDC_backbone_temp)

    ratio_base = 0

    for temp_test in range(i + 2):
        learned_scene = scenes_loop[temp_test]
        print(f"Testing on Scene: {learned_scene}")

        learned_val_data = val_scene_datasets[learned_scene]

        ratio = model.OOD_test(
            learned_val_data,
            params,
            MPDC_backbone_list,
            train_image_path=TRAIN_IMAGE_PATH,
            batch_size=BATCH_SIZE,
            device=None,
            dataset_name=DATASET_NAME,
            task_id=i,
            threshold=threshold
        )

        print('OOD detection ratio for ' + scenes[temp_test] + ': ' + str(ratio))

        if temp_test < (i + 1):
            ratio_base += (ratio / (i + 1))

    ratio_base += 0.05
    print('OOD detection threshold for ' + scenes[temp_test] + ': ' + str(ratio_base))

    if ratio_base < ratio:
        print('Novel motion pattern detected, model switching to accommodation phase.')
    else:
        print('Novel motion mode detection failed')

