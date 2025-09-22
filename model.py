import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.image_utils import get_patch
from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.preprocessing import augment_data, create_images_dict, augment_data_random, augment_data_select, augment_data_squence
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from test import evaluate, evaluate_inference
from train import train, get_feature
import cluster_loss
import cv2
import os
import math

class LoRAConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        rank: int = 8,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride, padding=padding,
            bias=bias
        )
        for p in self.conv.parameters():
            p.requires_grad = False

        self.lora_downs = nn.ModuleList()
        self.lora_ups   = nn.ModuleList()
        self.alphas     = []
        self.rank       = rank

        self._add_new_lora_pair(rank, alpha)

    def forward(self, x: torch.Tensor, task_id: int, lora_enabled: bool):
        out = self.conv(x)
        if lora_enabled:
            down = self.lora_downs[task_id](x)
            up   = self.lora_ups[task_id](down)
            scale = self.alphas[task_id] / self.rank
            out = out + scale * up
        return out

    # Add a LoRA group
    def _add_new_lora_pair(self, rank: int, alpha: float):
        down = nn.Conv2d(self.conv.in_channels, rank, 1, bias=False)
        up   = nn.Conv2d(rank, self.conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)

        self.lora_downs.append(down)
        self.lora_ups.append(up)
        self.alphas.append(alpha)

    def add_new_lora_pair(self, rank: int = 4, alpha: float = 1.0):
        self._add_new_lora_pair(rank, alpha)

class MaskedYNetLoRAEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels = (64, 128, 256, 512, 512),
        lora_enabled: bool = False,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lora_enabled = lora_enabled
        self.rank   = rank
        self.alpha  = alpha
        self.stages = nn.ModuleList()

        # Conv + ReLU
        self.stages.append(
            nn.Sequential(
                LoRAConv2d(in_channels, channels[0], 3, 1, 1, bias=False,
                           rank=rank, alpha=alpha),
                nn.ReLU(inplace=True),
            )
        )

        # MaxPool + 2×(Conv + ReLU)
        for i in range(len(channels) - 1):
            self.stages.append(
                nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    LoRAConv2d(channels[i],   channels[i+1], 3, 1, 1, bias=False,
                               rank=rank, alpha=alpha),
                    nn.ReLU(inplace=True),
                    LoRAConv2d(channels[i+1], channels[i+1], 3, 1, 1, bias=False,
                               rank=rank, alpha=alpha),
                    nn.ReLU(inplace=True),
                )
            )

        self.stages.append(nn.Sequential(nn.MaxPool2d(2, 2)))

    def forward(self, x: torch.Tensor, task_id: int = 0):

        features = []
        for stage in self.stages:
            for layer in stage:
                if isinstance(layer, LoRAConv2d):
                    x = layer(x, task_id, self.lora_enabled)
                else:
                    x = layer(x)
            features.append(x)
        return features

    # LoRA identifier
    def set_mask_enabled(self, enabled: bool):
        self.lora_enabled = enabled

    # Add a LoRA group
    def update_lora(self, rank: int = None, alpha: float = None):
        rank  = rank  if rank  is not None else self.rank
        alpha = alpha if alpha is not None else self.alpha

        for stage in self.stages:
            for layer in stage:
                if isinstance(layer, LoRAConv2d):
                    layer.add_new_lora_pair(rank, alpha)

class YNetDecoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
		super(YNetDecoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if traj:
			encoder_channels = [channel+traj for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		decoder_channels = decoder_channels

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)


		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature)
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x


class YNetTorch(nn.Module):
	def __init__(self, obs_len, pred_len, segmentation_model_fp, use_features_only=False, semantic_classes=6,
				 encoder_channels=[], decoder_channels=[], waypoints=1):
		super(YNetTorch, self).__init__()

		if segmentation_model_fp is not None:
			self.semantic_segmentation = torch.load(segmentation_model_fp)
			if use_features_only:
				self.semantic_segmentation.segmentation_head = nn.Identity()
				semantic_classes = 16  # instead of classes use number of feature_dim
		else:
			self.semantic_segmentation = nn.Identity()


		self.encoder = MaskedYNetLoRAEncoder(in_channels=semantic_classes + obs_len, channels=encoder_channels)

		self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
		self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=waypoints)

		self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

	def segmentation(self, image):
		return self.semantic_segmentation(image)

	# Forward pass for goal decoder
	def pred_goal(self, features):
		goal = self.goal_decoder(features)
		return goal

	# Forward pass for trajectory decoder
	def pred_traj(self, features):
		traj = self.traj_decoder(features)
		return traj

	# Forward pass for feature encoder, returns list of feature maps
	def pred_features(self, x, task_id):
		features = self.encoder(x, task_id)
		return features

	# Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
	def softmax(self, x):
		return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

	# Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
	def softargmax(self, output):
		return self.softargmax_(output)

	def sigmoid(self, output):
		return torch.sigmoid(output)

	def softargmax_on_softmax_map(self, x):
		pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
		pos_x = pos_x.reshape(-1)
		pos_y = pos_y.reshape(-1)
		x = x.flatten(2)

		estimated_x = pos_x * x
		estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
		estimated_y = pos_y * x
		estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
		softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
		return softargmax_coords


class YNet:
	def __init__(self, obs_len, pred_len, params):
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.division_factor = 2 ** len(params['encoder_channels'])

		self.model = YNetTorch(obs_len=obs_len,
							   pred_len=pred_len,
							   segmentation_model_fp=params['segmentation_model_fp'],
							   use_features_only=params['use_features_only'],
							   semantic_classes=params['semantic_classes'],
							   encoder_channels=params['encoder_channels'],
							   decoder_channels=params['decoder_channels'],
							   waypoints=len(params['waypoints']))

	def train(self, train_data, repaly_data, val_data, params, train_image_path, val_image_path, experiment_name,
			  batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None, task_id=0, replay=False):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Disable LoRA
		self.model.encoder.set_mask_enabled(False)

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		self.homo_mat = None
		seg_mask = False

		train_images = {}
		replay_images = {}

		# Load train images and augment train data and images
		if task_id > 0 and replay:
			df_replay, replay_images = augment_data(repaly_data, image_path=train_image_path, image_file=image_file_name,
												  seg_mask=seg_mask)
		else:
			df_train, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
												  seg_mask=seg_mask)

		# Load val scene images
		val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name)

		# Initialize dataloaders
		if task_id > 0 and replay:
			replay_dataset = SceneDataset(df_replay, resize=params['resize'], total_len=total_len)
			replay_loader = DataLoader(replay_dataset, batch_size=batch_size, collate_fn=scene_collate, shuffle=True)
		else:
			train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
			train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=scene_collate, shuffle=True)

		val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
		val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		if task_id > 0 and replay:
			resize(replay_images, factor=params['resize'], seg_mask=seg_mask)
			pad(replay_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
			preprocess_image_for_segmentation(replay_images, seg_mask=seg_mask)
		else:
			resize(train_images, factor=params['resize'], seg_mask=seg_mask)
			pad(train_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
			preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

		resize(val_images, factor=params['resize'], seg_mask=seg_mask)
		pad(val_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

		model = self.model.to(device)

		for p in model.parameters():
			p.requires_grad = True

		# Freeze segmentation model
		for param in model.semantic_segmentation.parameters():
			param.requires_grad = False

		epochs = params['num_epochs']

		# Except for initial training, keep the encoder frozen
		if replay or task_id > 0:
			decoder_params = list(model.goal_decoder.parameters()) + \
							list(model.traj_decoder.parameters())
			optimizer = torch.optim.Adam(decoder_params, lr=params["learning_rate"])
		else:
			optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

		criterion = nn.BCEWithLogitsLoss()

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
		gt_template = torch.Tensor(gt_template).to(device)

		best_test_ADE = 99999999999999

		self.train_ADE = []
		self.train_FDE = []
		self.val_ADE = []
		self.val_FDE = []

		if not replay:
			print('Start training')
			for e in tqdm(range(epochs), desc='Epoch'):
				train_ADE, train_FDE, train_loss = train(model, train_loader, train_images, e, obs_len, pred_len,
														 batch_size, params, gt_template, device,
														 input_template, optimizer, criterion, dataset_name, self.homo_mat, task_id)
				self.train_ADE.append(train_ADE)
				self.train_FDE.append(train_FDE)

				val_ADE, val_FDE = evaluate(model, val_loader, val_images, num_goals, num_traj,
											obs_len=obs_len, batch_size=1,
											device=device, input_template=input_template,
											waypoints=params['waypoints'], resize=params['resize'],
											temperature=params['temperature'], use_TTST=False,
											use_CWS=False, dataset_name=dataset_name,
											homo_mat=self.homo_mat, mode='val', task_id=task_id)
				print(f'Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
				self.val_ADE.append(val_ADE)
				self.val_FDE.append(val_FDE)

				if val_ADE < best_test_ADE:
					print(f'Best Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
					torch.save(model.state_dict(), 'trained_models/' + experiment_name + '_weights.pt')
					best_test_ADE = val_ADE

		else:
			print('Start replaying')
			for e in tqdm(range(epochs), desc='Epoch'):
				train_ADE, train_FDE, train_loss = train(model, replay_loader, replay_images, e, obs_len, pred_len,
														 batch_size, params, gt_template, device,
														 input_template, optimizer, criterion, dataset_name, self.homo_mat, task_id)
				self.train_ADE.append(train_ADE)
				self.train_FDE.append(train_FDE)

				val_ADE, val_FDE = evaluate(model, val_loader, val_images, num_goals, num_traj,
											obs_len=obs_len, batch_size=1,
											device=device, input_template=input_template,
											waypoints=params['waypoints'], resize=params['resize'],
											temperature=params['temperature'], use_TTST=False,
											use_CWS=False, dataset_name=dataset_name,
											homo_mat=self.homo_mat, mode='val', task_id=task_id)
				print(f'Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
				self.val_ADE.append(val_ADE)
				self.val_FDE.append(val_FDE)

				if val_ADE < best_test_ADE:
					print(f'Best Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
					torch.save(model.state_dict(), 'trained_models/' + experiment_name + '_weights.pt')
					best_test_ADE = val_ADE

	def evaluate_inference(self, data, params, image_path, batch_size=8, num_goals=20, num_traj=1, rounds=1,
						   device=None, dataset_name=None, CL=False, task_id=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Disable LoRA
		self.model.encoder.set_mask_enabled(False)

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		self.homo_mat = None
		seg_mask = False


		test_images = {}
		for scene in data[2]:
			if image_file_name == 'oracle.png':
				im = cv2.imread(os.path.join(image_path, scene, image_file_name), 0)
			else:
				im = cv2.imread(os.path.join(image_path, scene, image_file_name))
			test_images[scene] = im

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(test_images, factor=params['resize'], seg_mask=seg_mask)
		pad(test_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
		preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)

		model = self.model.to(device)

		# Create template
		size = int(4200 * params['resize'])

		input_template = torch.Tensor(create_dist_mat(size=size)).to(device)

		self.eval_ADE = []
		self.eval_FDE = []

		print('Start testing')
		for e in tqdm(range(rounds), desc='Round'):
			final_list, path_list, scene_list, test_ADE, test_FDE = evaluate_inference(model, data, test_images, num_goals, num_traj,
										  obs_len=self.obs_len, batch_size=batch_size,
										  device=device, input_template=input_template,
										  waypoints=params['waypoints'], resize=params['resize'],
										  temperature=params['temperature'], use_TTST=True,
										  use_CWS=True if len(params['waypoints']) > 1 else False,
										  rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'],
										  dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test', task_id=task_id)
			print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

			self.eval_ADE.append(test_ADE)
			self.eval_FDE.append(test_FDE)

		print(f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest FDE: {sum(self.eval_FDE) / len(self.eval_FDE)}')

		if CL:
			return final_list, path_list, scene_list, sum(self.eval_ADE) / len(self.eval_ADE), sum(self.eval_FDE) / len(self.eval_FDE)


	def load(self, path):
		print(self.model.load_state_dict(torch.load(path)))

	def save(self, path):
		torch.save(self.model.state_dict(), path)

	def train_MPDC(self, train_data, repaly_data, val_data, params, MPDC_module, train_image_path, val_image_path,
				  experiment_name, batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None, task_id=0):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Activate LoRA
		self.model.encoder.set_mask_enabled(True)

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		self.homo_mat = None
		seg_mask = False

		model = self.model.to(device)
		MPDC_network = MPDC_module.to(device)

		# Initialize the contrast losses
		criterion_instance = cluster_loss.ProtoContrastLoss(params['batch_OOD'], 0.5).to(device)
		criterion_cluster = cluster_loss.ClusterLoss((0 + 1) * params['class_scene'], 1.0).to(device)

		epochs = params['num_epochs_OOD']

		# Only train LoRA and MPDC
		for p in model.parameters():
			p.requires_grad = False

		for name, p in model.named_parameters():
			if ("lora_downs" in name or "lora_ups" in name):
				p.requires_grad = True

		lora_params = [p for p in model.parameters() if p.requires_grad]
		ood_params = [p for p in MPDC_network.parameters() if p.requires_grad]

		optimizer = torch.optim.Adam([
			{'params': lora_params, 'lr': params["learning_rate"]},
			{'params': ood_params, 'lr': params["learning_rate"]}
		])

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		print('OOD training')
		for e in tqdm(range(epochs), desc='Epoch'):

			train_images1 = {}

			# Load train images and augment train data and images
			df_train1, train_images1 = augment_data_random(train_data, image_path=train_image_path, image_file=image_file_name,
												  seg_mask=seg_mask)

			# Initialize dataloaders
			train_dataset1 = SceneDataset(df_train1, resize=params['resize'], total_len=total_len)
			train_loader1 = DataLoader(train_dataset1, batch_size=1, collate_fn=scene_collate, shuffle=False)

			# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
			resize(train_images1, factor=params['resize'], seg_mask=seg_mask)
			pad(train_images1,
				division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
			preprocess_image_for_segmentation(train_images1, seg_mask=seg_mask)

			train_images2 = {}

			# Load train images and augment train data and images
			df_train2, train_images2 = augment_data_random(train_data, image_path=train_image_path,
														   image_file=image_file_name,
														   seg_mask=seg_mask)

			# Initialize dataloaders
			train_dataset2 = SceneDataset(df_train2, resize=params['resize'], total_len=total_len)
			train_loader2 = DataLoader(train_dataset2, batch_size=1, collate_fn=scene_collate, shuffle=False)


			# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
			resize(train_images2, factor=params['resize'], seg_mask=seg_mask)
			pad(train_images2,
				division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
			preprocess_image_for_segmentation(train_images2, seg_mask=seg_mask)

			loss_epoch_instance = []
			loss_epoch_cluster = []
			# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
			for ((trajectory1, meta1, scene1), (trajectory2, meta2, scene2)) in zip(train_loader1, train_loader2):
				scene_image1 = train_images1[scene1].to(device).unsqueeze(0)
				scene_image1 = model.segmentation(scene_image1)

				scene_image2 = train_images2[scene2].to(device).unsqueeze(0)
				scene_image2 = model.segmentation(scene_image2)
				
				# inner loop, for each trajectory in the scene
				for i in range(0, len(trajectory1), batch_size):

					optimizer.zero_grad()

					# Create Heatmaps for past and ground-truth future trajectories
					_, _, H, W = scene_image1.shape  # image shape

					observed1 = trajectory1[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
					observed_map1 = get_patch(input_template, observed1, H, W)
					observed_map1 = torch.stack(observed_map1).reshape([-1, obs_len, H, W])

					# Concatenate heatmap and semantic map
					semantic_map1 = scene_image1.expand(observed_map1.shape[0], -1, -1, -1)  # expand to match heatmap size
					feature_input1 = torch.cat([semantic_map1, observed_map1], dim=1)

					_, _, H, W = scene_image2.shape  # image shape

					observed2 = trajectory2[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
					observed_map2 = get_patch(input_template, observed2, H, W)
					observed_map2 = torch.stack(observed_map2).reshape([-1, obs_len, H, W])

					# Concatenate heatmap and semantic map
					semantic_map2 = scene_image2.expand(observed_map2.shape[0], -1, -1, -1)  # expand to match heatmap size
					feature_input2 = torch.cat([semantic_map2, observed_map2], dim=1)

					# feature_input = torch.cat((feature_input1, feature_input2), dim=0)

					# Forward pass
					# Calculate features
					feature1 = model.pred_features(feature_input1, task_id)

					feature1 = feature1[5].reshape(batch_size, -1)

					# Fill in the dimension size
					pad_size = 12288 - feature1.size(1)

					feature1 = F.pad(feature1, (0, pad_size), mode='constant', value=0)

					feature2 = model.pred_features(feature_input2, task_id)

					feature2 = feature2[5].reshape(batch_size, -1)
					pad_size = 12288 - feature2.size(1)

					feature2 = F.pad(feature2, (0, pad_size), mode='constant', value=0)

					proj_feat1, proj_feat2, z_i, z_j, c_i, c_j, mu = MPDC_network(feature1, feature2, 0, device)
					loss_instance = criterion_instance(proj_feat1, proj_feat2, z_i, z_j, c_i, mu, e)
					loss_cluster = criterion_cluster(c_i, c_j)

					loss = loss_instance + loss_cluster
					loss.backward()
					optimizer.step()

					loss_epoch_instance.append(loss_instance.item())
					loss_epoch_cluster.append(loss_cluster.item())

			print(f'Epoch {e}: {torch.tensor(loss_epoch_instance).mean()}, {torch.tensor(loss_epoch_cluster).mean()}')

		torch.save(model.state_dict(), 'trained_models/' + experiment_name + '_weights.pt')

		return MPDC_network

	def replay_select(self, train_data, params, MPDC_module, train_image_path, val_image_path,
				  experiment_name, batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None,
				  task_id=0):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Activate LoRA
		self.model.encoder.set_mask_enabled(True)

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		self.homo_mat = None
		seg_mask = False

		model = self.model.to(device)
		MPDC_network = MPDC_module.to(device)

		model.eval()

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'],
													   normalize=False)
		gt_template = torch.Tensor(gt_template).to(device)

		train_images = {}
		replay_images = {}

		# Load train images and augment train data and images
		df_train, train_images = augment_data_select(train_data, image_path=train_image_path,
													   image_file=image_file_name,
													   seg_mask=seg_mask)

		# Initialize dataloaders
		train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=scene_collate,
								   shuffle=False)


		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(train_images, factor=params['resize'], seg_mask=seg_mask)
		pad(train_images,
			division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

		feature_all = get_feature(model, train_loader, train_images, obs_len, pred_len,
									batch_size, params, gt_template, device,
									input_template, None, dataset_name, self.homo_mat, task_id)

		z, c, mu = MPDC_network.forward_cluster(feature_all, 0, device)

		results = cluster_loss.compute_sorted_cosine_similarities(z, c, mu)

		return results

	def OOD_update(self, train_data, replay_data, params, MPDC_backbone, train_image_path, batch_size=8, device=None,
				   dataset_name=None, task_id=0, prototype=None, covariance=None):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Activate LoRA
		self.model.encoder.set_mask_enabled(True)

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		self.homo_mat = None
		seg_mask = False

		model = self.model.to(device)
		MPDC_backbone = MPDC_backbone.to(device)

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'],
													   normalize=False)
		gt_template = torch.Tensor(gt_template).to(device)

		all_cosine_sim = [[] for _ in range(train_data.shape[0] // 20)]
		all_preds = [[] for _ in range(train_data.shape[0] // 20)]
		for temp in range(8):

			train_images = {}

			# Load train images and augment train data and images
			df_train, train_images = augment_data_squence(train_data, image_path=train_image_path,
														  image_file=image_file_name,
														  seg_mask=seg_mask, n_aug=temp)

			# Initialize dataloaders
			train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
			train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=scene_collate,
									  shuffle=False)

			# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
			resize(train_images, factor=params['resize'], seg_mask=seg_mask)
			pad(train_images,
				division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
			preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

			feature_all = get_feature(model, train_loader, train_images, obs_len, pred_len,
									  batch_size, params, gt_template, device,
									  input_template, None, dataset_name, self.homo_mat, task_id)

			cosine_sim, c = MPDC_backbone.forward_score(feature_all, task_id, prototype, covariance, device)

			for i, (sim, pred) in enumerate(zip(cosine_sim, c)):
				all_cosine_sim[i].append(sim.item())
				all_preds[i].append(pred.item())

		# Post-processing: Statistical prediction of categories and calculation of effective average similarity
		final_cosine_sim = []
		final_pred_classes = []

		for preds, sims in zip(all_preds, all_cosine_sim):
			pred_counts = {}
			for cls in preds:
				pred_counts[cls] = pred_counts.get(cls, 0) + 1

			# OOD Criteria 1
			majority_class = None
			for cls, count in pred_counts.items():
				if count >= 5:
					majority_class = cls
					break

			if majority_class is not None:
				valid_sims = [sim for sim, cls in zip(sims, preds) if cls == majority_class]
				avg_sim = sum(valid_sims) / len(valid_sims)
				final_cosine_sim.append(avg_sim)
				final_pred_classes.append(majority_class)
			else:
				final_cosine_sim.append(0.0)
				final_pred_classes.append(-1)

		final_cosine_sim = torch.tensor(final_cosine_sim).to(device)
		c = torch.tensor(final_pred_classes).to(device)

		num_clusters = params['class_scene']
		cluster_70th_scores = torch.zeros(num_clusters).to(device)

		for i in range(num_clusters):
			cluster_mask = (c == i)
			if cluster_mask.sum() > 0:
				# OOD Criteria 2
				cluster_similarities = final_cosine_sim[cluster_mask]

				sorted_similarities, _ = torch.sort(cluster_similarities, descending=True)

				k = max(0, int(len(sorted_similarities) * 0.7) - 1)

				if sorted_similarities[k] > 0:
					cluster_70th_scores[i] = sorted_similarities[k]
				else:
					cluster_70th_scores[i] = 0.0
			else:
				cluster_70th_scores[i] = 1.0

		return (cluster_70th_scores).detach().cpu()

	def OOD_test(self, train_data, params, MPDC_backbone_list, train_image_path, batch_size=8, device=None, dataset_name=None,
				  task_id=0, threshold=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Activate LoRA
		self.model.encoder.set_mask_enabled(True)

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		self.homo_mat = None
		seg_mask = False

		model = self.model.to(device)

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'],
													   normalize=False)
		gt_template = torch.Tensor(gt_template).to(device)

		# Initialize the index, all samples start in the low-confidence set.
		ratio = 1.0
		low_confidence_indices = torch.arange(train_data.shape[0]//20, device=device)

		for temp_1 in range(task_id + 1):
			MPDC_network = MPDC_backbone_list[temp_1].to(device)
			all_cosine_sim = [[] for _ in range(len(low_confidence_indices))]
			all_preds = [[] for _ in range(len(low_confidence_indices))]

			for temp_2 in range(8):
				train_images = {}

				# Load train images and augment train data and images
				df_train, train_images = augment_data_squence(train_data, image_path=train_image_path,
															image_file=image_file_name,
															seg_mask=seg_mask, n_aug=temp_2)

				# Initialize dataloaders
				train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
				train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=scene_collate,
										shuffle=False)

				# Preprocess images for segmentation model
				resize(train_images, factor=params['resize'], seg_mask=seg_mask)
				pad(train_images, division_factor=self.division_factor)
				preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

				# Extract features
				feature_all = get_feature(model, train_loader, train_images, obs_len, pred_len,
										batch_size, params, gt_template, device,
										input_template, None, dataset_name, self.homo_mat, temp_1)

				# Compute cosine similarity
				cosine_sim, c = MPDC_network.forward_threshold(feature_all[low_confidence_indices], task_id, threshold[temp_1], device)

				# Store the results
				for i, (sim, pred) in enumerate(zip(cosine_sim, c)):
					all_cosine_sim[i].append(sim.item())
					all_preds[i].append(pred.item())

			# Post-processing: Statistical prediction of categories and calculation of effective average similarity
			final_cosine_sim = []
			final_pred_classes = []

			for preds, sims in zip(all_preds, all_cosine_sim):
				pred_counts = {}
				for cls in preds:
					pred_counts[cls] = pred_counts.get(cls, 0) + 1

				# OOD Criteria 1
				majority_class = None
				for cls, count in pred_counts.items():
					if count >= 5:
						majority_class = cls
						break

				if majority_class is not None:
					valid_sims = [sim for sim, cls in zip(sims, preds) if cls == majority_class]
					avg_sim = sum(valid_sims) / len(valid_sims)
					final_cosine_sim.append(avg_sim)
					final_pred_classes.append(majority_class)
				else:
					final_cosine_sim.append(0.0)
					final_pred_classes.append(0)

			final_cosine_sim = torch.tensor(final_cosine_sim).to(device)
			c = torch.tensor(final_pred_classes).to(device)

			sample_thresholds = threshold[temp_1][c].to(device)

			# OOD Criteria 2
			mask_exceed = final_cosine_sim >= sample_thresholds
			proportion_exceed = mask_exceed.float().mean()

			# Update low-confidence indices — Retain only indices that still fail to meet the threshold in the current round
			low_confidence_indices = low_confidence_indices[~mask_exceed]

			ratio *= (1 - proportion_exceed)
			if ratio > params['OOD_threshold']:
				continue
			else:
				break

		return low_confidence_indices.shape[0] / (train_data.shape[0] // 20)























