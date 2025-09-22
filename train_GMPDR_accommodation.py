import pandas as pd
import yaml
import argparse
import torch
from model import YNet
import numpy as np
import pickle
from MPDC_head import Network

CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = 'data/SDD/train_trajnet.pkl'
TRAIN_IMAGE_PATH = 'data/SDD/train'
VAL_DATA_PATH = 'data/SDD/test.pickle'
VAL_IMAGE_PATH = 'data/SDD/test'
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
    "gates": ["gates_0", "gates_1", "gates_3", "gates_4", "gates_5", "gates_6", "gates_7", "gates_8"],
    "hyang": ["hyang_4", "hyang_5", "hyang_6", "hyang_7", "hyang_9"],
    "nexus": ["nexus_0", "nexus_1", "nexus_2", "nexus_3", "nexus_4", "nexus_7", "nexus_8", "nexus_9"],
    "coupa": ["coupa_3"]
}

scene_column_index = 4
group_size = 20
train_scene_datasets = {}
val_scene_datasets = {}

# Define the training set and validation set
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

for scene in train_scene_datasets:
    print(f"Scene: {scene}, Train Shape: {train_scene_datasets[scene].shape}, Val Shape: {val_scene_datasets[scene].shape}")

# Initialize the model
scenes = ['gates', 'hyang', 'nexus', 'coupa']
model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
MPDC_module = Network(12288, 512, 128, params['class_scene'])
scenes_loop = list(train_scene_datasets.keys())
incremental_results = {}
repaly_sample = None

# Training and Testing (Accommodation)
for i, current_scene in enumerate(scenes_loop):

    print(f"-------------------Training on Scene: {current_scene}----------------------")

    if i > 0:
        load_path_nsp = 'trained_models/' + scenes[i-1] + '_weights.pt'
        model.load(load_path_nsp)

        # Define new Lora and MPDC module
        model.model.encoder.update_lora()
        MPDC_module = Network(12288, 512, 128, params['class_scene'])

    current_train_data = train_scene_datasets[current_scene]
    current_val_data = val_scene_datasets[current_scene]

    # Train the backbone
    model.train(
        current_train_data,
        repaly_sample,
        current_val_data,
        params,
        train_image_path=TRAIN_IMAGE_PATH,
        val_image_path=TRAIN_IMAGE_PATH,
        experiment_name=current_scene,
        batch_size=BATCH_SIZE,
        num_goals=NUM_GOALS,
        num_traj=NUM_TRAJ,
        device=None,
        dataset_name=DATASET_NAME,
        task_id = i,
        replay = False
    )


    load_path_nsp = 'trained_models/' + scenes[i] + '_weights.pt'
    model.load(load_path_nsp)

    # Train the MPDC_module
    MPDC_module = model.train_MPDC(
                    current_train_data,
                    repaly_sample,
                    current_val_data,
                    params,
                    MPDC_module,
                    train_image_path=TRAIN_IMAGE_PATH,
                    val_image_path=TRAIN_IMAGE_PATH,
                    experiment_name=current_scene,
                    batch_size=params['batch_OOD'],
                    num_goals=NUM_GOALS,
                    num_traj=NUM_TRAJ,
                    device=None,
                    dataset_name=DATASET_NAME,
                    task_id=i
    )
    torch.save(MPDC_module.state_dict(), 'trained_models/' + scenes[i] + '_MPDC_weights.pt')

    # Select representative sparse replay samples
    results = model.replay_select(
        current_train_data,
        params,
        MPDC_module,
        train_image_path=TRAIN_IMAGE_PATH,
        val_image_path=TRAIN_IMAGE_PATH,
        experiment_name=current_scene,
        batch_size=1,
        num_goals=NUM_GOALS,
        num_traj=NUM_TRAJ,
        device=None,
        dataset_name=DATASET_NAME,
        task_id=i
    )

    index_all = torch.tensor([])
    if i == 0:
        for temp in range(params["class_scene"]):
            try:
                num = results[temp]['sorted_original_indices'].size(0)
                select_num = int(num // 100 + 1)
                permuted_indices = torch.randperm(num)[:select_num]
                index = results[temp]['sorted_original_indices'][permuted_indices]
                index_all = torch.cat((index_all, index))
            except:
                continue

        group_indices = index_all.numpy()
        selected_rows = np.concatenate([np.arange(g * 20, (g + 1) * 20) for g in group_indices])
        df_selected = current_train_data.iloc[selected_rows]
        repaly_sample = df_selected

    else:
        for temp in range(params["class_scene"]*(i+1)):
            try:
                num = results[temp]['sorted_original_indices'].size(0)
                select_num = int(num // 100 + 1)
                permuted_indices = torch.randperm(num)[:select_num]
                index = results[temp]['sorted_original_indices'][permuted_indices]
                index_all = torch.cat((index_all, index))
            except:
                continue

        group_indices = index_all.numpy()
        selected_rows = np.concatenate([np.arange(g * 20, (g + 1) * 20) for g in group_indices])
        df_selected = current_train_data.iloc[selected_rows]
        repaly_sample = pd.concat([repaly_sample, df_selected], ignore_index=True)

    # Replay
    load_path_nsp = 'trained_models/' + scenes[i] + '_weights.pt'
    model.load(load_path_nsp)
    if i > 0:
        print(f"Scene: {current_scene}, Replay Data Shape: {repaly_sample.shape}")
        model.train(
            current_train_data,
            repaly_sample,
            current_val_data,
            params,
            train_image_path=TRAIN_IMAGE_PATH,
            val_image_path=TRAIN_IMAGE_PATH,
            experiment_name=current_scene,
            batch_size=BATCH_SIZE,
            num_goals=NUM_GOALS,
            num_traj=NUM_TRAJ,
            device=None,
            dataset_name=DATASET_NAME,
            task_id=i,
            replay=True
        )

    # Testing learned scenarios
    print(f"---------------Evaluating on all learned scenes up to {current_scene}-------------")
    ADE_results = {}
    FDE_results = {}
    save_goal = None

    load_path_nsp = 'trained_models/' + scenes[i] + '_weights.pt'
    model.load(load_path_nsp)

    for temp_test in range(len(scenes_loop[: i + 1])):
        learned_scene = scenes_loop[temp_test]
        print(f"Testing on Scene: {learned_scene}")
        with open(VAL_DATA_PATH, 'rb') as f:
            test_data_all = pickle.load(f)
        test_data = [[],[],[]]
        for temp in range(len(test_data_all[2])):
            if scenes[temp_test] in test_data_all[2][temp]:
                test_data[0].append(test_data_all[0][temp] * params["resize"])
                test_data[1].append(test_data_all[1][temp])
                test_data[2].append(test_data_all[2][temp])

        final_list, path_list, scene_list, ADE, FDE = model.evaluate_inference(
            test_data,
            params,
            image_path=VAL_IMAGE_PATH,
            batch_size=1,
            rounds=ROUNDS,
            num_goals=NUM_GOALS,
            num_traj=NUM_TRAJ,
            device=None,
            dataset_name=DATASET_NAME,
            CL=True,
            task_id=i
        )

        # Store test results
        ADE_results[learned_scene] = ADE
        FDE_results[learned_scene] = FDE

        if temp_test == 0:
            save_goal = [final_list, path_list, scene_list]
        else:
            save_goal[0].extend(final_list)
            save_goal[1].extend(path_list)
            save_goal[2].extend(scene_list)

    incremental_results[current_scene] = {
        "ADE": ADE_results,
        "FDE": FDE_results
    }

    print("Incremental Learning Results:")
    for scene, results in incremental_results.items():
        print(f"Scene: {scene}")
        for metric, values in results.items():
            print(f"  {metric}:")
            for learned_scene, score in values.items():
                print(f"    {learned_scene}: {score}")

scene_names = list(incremental_results.keys())
num_scenes = len(scene_names)

ADE_matrix = np.zeros((num_scenes, num_scenes))
FDE_matrix = np.zeros((num_scenes, num_scenes))

for i, scene in enumerate(scene_names):
    ade_results = incremental_results[scene]["ADE"]
    fde_results = incremental_results[scene]["FDE"]
    
    for j, target_scene in enumerate(scene_names[:i+1]):
        ADE_matrix[i, j] = ade_results.get(target_scene, 0)
        FDE_matrix[i, j] = fde_results.get(target_scene, 0)


final_ADE = ADE_matrix[-1]
final_FDE = FDE_matrix[-1]

ADE_final_avg = np.mean(final_ADE[final_ADE != 0])
FDE_final_avg = np.mean(final_FDE[final_FDE != 0])

print(f"Final ADE average: {ADE_final_avg}")
print(f"Final FDE average: {FDE_final_avg}")


ADE_step_avg = []
FDE_step_avg = []

for i in range(len(ADE_matrix)):
    ade_values = ADE_matrix[i][:i+1]
    fde_values = FDE_matrix[i][:i+1]
    
    ADE_step_avg.append(np.mean(ade_values))
    FDE_step_avg.append(np.mean(fde_values))

ADE_overall_avg = np.mean(ADE_step_avg)
FDE_overall_avg = np.mean(FDE_step_avg)

print(f"Incremental ADE average: {ADE_overall_avg}")
print(f"Incremental FDE average: {FDE_overall_avg}")


initial_ADE = np.diag(ADE_matrix)
initial_FDE = np.diag(FDE_matrix)

final_ADE_all = ADE_matrix[-1]
final_FDE_all = FDE_matrix[-1]

valid_indices = initial_ADE != 0

ADE_forgetting = initial_ADE[:-1] - final_ADE_all[:-1]
FDE_forgetting = initial_FDE[:-1] - final_FDE_all[:-1]

print(f"ADE Forgetting Rates: {ADE_forgetting}")
print(f"FDE Forgetting Rates: {FDE_forgetting}")
print(f"Average ADE Forgetting: {np.mean(ADE_forgetting)}")
print(f"Average FDE Forgetting: {np.mean(FDE_forgetting)}")

