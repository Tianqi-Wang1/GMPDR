import torch
import torch.nn as nn
from utils.image_utils import get_patch, sampling, image2world
from utils.kmeans import kmeans


def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
	ax = torch.linspace(0, H, H, device=device) - coordinates[1]
	ay = torch.linspace(0, W, W, device=device) - coordinates[0]
	xx, yy = torch.meshgrid([ax, ay])
	meshgrid = torch.stack([yy, xx], dim=-1)
	radians = torch.atan2(dist[0], dist[1])

	c, s = torch.cos(radians), torch.sin(radians)
	R = torch.Tensor([[c, s], [-s, c]]).to(device)
	if rot:
		R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
	dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero

	conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
	conv = torch.square(conv)
	T = torch.matmul(R, conv)
	T = torch.matmul(T, R.T)

	kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
	kernel = torch.exp(-0.5 * kernel)
	return kernel / kernel.sum()


def evaluate(model, val_loader, val_images, num_goals, num_traj, obs_len, batch_size, device, input_template,
			 waypoints, resize, temperature, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None,
			 dataset_name=None, homo_mat=None, mode='val', task_id=None):
	model.eval()
	val_ADE = []
	val_FDE = []
	with torch.no_grad():
		# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
		for trajectory, meta, scene in val_loader:
			# Get scene image and apply semantic segmentation
			scene_image = val_images[scene].to(device).unsqueeze(0)
			scene_image = model.segmentation(scene_image)

			for i in range(0, len(trajectory), batch_size):
				# Create Heatmaps for past and ground-truth future trajectories
				_, _, H, W = scene_image.shape
				observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
				observed_map = get_patch(input_template, observed, H, W)
				observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

				gt_future = trajectory[i:i+batch_size, obs_len:].to(device)
				semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)

				# Forward pass
				# Calculate features
				feature_input = torch.cat([semantic_image, observed_map], dim=1)
				features = model.pred_features(feature_input, task_id)

				# Predict goal and waypoint probability distributions
				pred_waypoint_map = model.pred_goal(features)
				pred_waypoint_map = pred_waypoint_map[:, waypoints]

				pred_waypoint_map_sigmoid = pred_waypoint_map / temperature
				pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)

				################################################ TTST ##################################################
				if use_TTST:
					# TTST Begin
					# sample a large amount of goals to be clustered
					goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
					goal_samples = goal_samples.permute(2, 0, 1, 3)

					num_clusters = num_goals - 1
					goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample

					# Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
					goal_samples_list = []
					for person in range(goal_samples.shape[1]):
						goal_sample = goal_samples[:, person, 0]

						# Actual k-means clustering, Outputs:
						# cluster_ids_x -  Information to which cluster_idx each point belongs to
						# cluster_centers - list of centroids, which are our new goal samples
						cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
						goal_samples_list.append(cluster_centers)

					goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
					goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
					# TTST End

				# Not using TTST
				else:
					goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=num_goals)
					goal_samples = goal_samples.permute(2, 0, 1, 3)

				# Predict waypoints:
				# in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
				if len(waypoints) == 1:
					waypoint_samples = goal_samples

				################################################ CWS ###################################################
				# CWS Begin
				if use_CWS and len(waypoints) > 1:
					sigma_factor = CWS_params['sigma_factor']
					ratio = CWS_params['ratio']
					rot = CWS_params['rot']

					goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
					last_observed = trajectory[i:i+batch_size, obs_len-1].to(device)  # [N, 2]
					waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
					for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
						waypoint_list = []  # for each K sample have a separate list
						waypoint_list.append(waypoint_samples)

						for waypoint_num in reversed(range(len(waypoints)-1)):
							distance = last_observed - waypoint_samples
							gaussian_heatmaps = []
							traj_idx = g_num // num_goals  # idx of trajectory for the same goal
							for dist, coordinate in zip(distance, waypoint_samples):  # for each person
								length_ratio = 1 / (waypoint_num + 2)
								gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
								sigma_factor_ = sigma_factor - traj_idx
								gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
							gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]

							waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
							waypoint_map = waypoint_map_before * gaussian_heatmaps
							# normalize waypoint map
							waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)

							# For first traj samples use softargmax
							if g_num // num_goals == 0:
								# Softargmax
								waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
								waypoint_samples = waypoint_samples.squeeze(0)
							else:
								waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
								waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
								waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
							waypoint_list.append(waypoint_samples)

						waypoint_list = waypoint_list[::-1]
						waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
						waypoint_samples_list.append(waypoint_list)
					waypoint_samples = torch.stack(waypoint_samples_list)

					# CWS End

				# If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
				elif not use_CWS and len(waypoints) > 1:
					waypoint_samples = sampling(pred_waypoint_map_sigmoid[:, :-1], num_samples=num_goals * num_traj)
					waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
					goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
					waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)

				# Interpolate trajectories given goal and waypoints
				future_samples = []
				for waypoint in waypoint_samples:
					waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
					waypoint_map = torch.stack(waypoint_map).reshape([-1, len(waypoints), H, W])

					waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
					waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

					traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]

					pred_traj_map = model.pred_traj(traj_input)
					pred_traj = model.softargmax(pred_traj_map)
					future_samples.append(pred_traj)
				future_samples = torch.stack(future_samples)

				gt_goal = gt_future[:, -1:]

				val_FDE.append(((((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
				val_ADE.append(((((gt_future - future_samples) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])

		val_ADE = torch.cat(val_ADE).mean()
		val_FDE = torch.cat(val_FDE).mean()

	return val_ADE.item(), val_FDE.item()


def evaluate_inference(model, data, val_images, num_goals, num_traj, obs_len, batch_size, device, input_template,
					   waypoints, resize, temperature, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None,
					   dataset_name=None, homo_mat=None, mode='val', task_id=None):

	model.eval()
	val_ADE = []
	val_FDE = []
	with torch.no_grad():
		# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
		# for trajectory, meta, scene in val_loader:
		final_list = []
		path_list =[]
		scene_list = []

		for temp in range(len(data[2])):
			trajectory = torch.from_numpy(data[0][temp])
			scene = data[2][temp]
			# Get scene image and apply semantic segmentation
			scene_image = val_images[scene].to(device).unsqueeze(0)
			scene_image = model.segmentation(scene_image)

			final = None
			path = None

			for i in range(0, len(trajectory), batch_size):
				# Create Heatmaps for past and ground-truth future trajectories
				_, _, H, W = scene_image.shape
				observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
				observed_map = get_patch(input_template, observed, H, W)
				observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

				gt_future = trajectory[i:i+batch_size, obs_len:].to(device)
				semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)

				# Forward pass
				# Calculate features
				feature_input = torch.cat([semantic_image, observed_map], dim=1)
				features = model.pred_features(feature_input, task_id)

				# Predict goal and waypoint probability distributions
				pred_waypoint_map = model.pred_goal(features)
				pred_waypoint_map = pred_waypoint_map[:, waypoints]

				pred_waypoint_map_sigmoid = pred_waypoint_map / temperature
				pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)

				################################################ TTST ##################################################
				if use_TTST:
					# TTST Begin
					# sample a large amount of goals to be clustered
					goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
					goal_samples = goal_samples.permute(2, 0, 1, 3)

					num_clusters = num_goals - 1
					goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample

					# Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
					goal_samples_list = []
					for person in range(goal_samples.shape[1]):
						goal_sample = goal_samples[:, person, 0]

						# Actual k-means clustering, Outputs:
						# cluster_ids_x -  Information to which cluster_idx each point belongs to
						# cluster_centers - list of centroids, which are our new goal samples
						cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
						goal_samples_list.append(cluster_centers)

					goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
					goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
					# TTST End

				# Not using TTST
				else:
					goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=num_goals)
					goal_samples = goal_samples.permute(2, 0, 1, 3)

				# Predict waypoints:
				# in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
				if len(waypoints) == 1:
					waypoint_samples = goal_samples

				################################################ CWS ###################################################
				# CWS Begin
				if use_CWS and len(waypoints) > 1:
					sigma_factor = CWS_params['sigma_factor']
					ratio = CWS_params['ratio']
					rot = CWS_params['rot']

					goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
					last_observed = trajectory[i:i+batch_size, obs_len-1].to(device)  # [N, 2]
					waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
					for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
						waypoint_list = []  # for each K sample have a separate list
						waypoint_list.append(waypoint_samples)

						for waypoint_num in reversed(range(len(waypoints)-1)):
							distance = last_observed - waypoint_samples
							gaussian_heatmaps = []
							traj_idx = g_num // num_goals  # idx of trajectory for the same goal
							for dist, coordinate in zip(distance, waypoint_samples):  # for each person
								length_ratio = 1 / (waypoint_num + 2)
								gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
								sigma_factor_ = sigma_factor - traj_idx
								gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
							gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]

							waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
							waypoint_map = waypoint_map_before * gaussian_heatmaps
							# normalize waypoint map
							waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)

							# For first traj samples use softargmax
							if g_num // num_goals == 0:
								# Softargmax
								waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
								waypoint_samples = waypoint_samples.squeeze(0)
							else:
								waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
								waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
								waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
							waypoint_list.append(waypoint_samples)

						waypoint_list = waypoint_list[::-1]
						waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
						waypoint_samples_list.append(waypoint_list)
					waypoint_samples = torch.stack(waypoint_samples_list)

					# CWS End

				# If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
				elif not use_CWS and len(waypoints) > 1:
					waypoint_samples = sampling(pred_waypoint_map_sigmoid[:, :-1], num_samples=num_goals * num_traj)
					waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
					goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
					waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)

				# Interpolate trajectories given goal and waypoints
				future_samples = []
				for waypoint in waypoint_samples:
					waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
					waypoint_map = torch.stack(waypoint_map).reshape([-1, len(waypoints), H, W])

					waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
					waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

					traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]

					pred_traj_map = model.pred_traj(traj_input)
					pred_traj = model.softargmax(pred_traj_map)
					future_samples.append(pred_traj)
				future_samples = torch.stack(future_samples)

				gt_goal = gt_future[:, -1:]

				val_FDE.append(((((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
				val_ADE.append(((((gt_future - future_samples) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])

				index = ((((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[1]
				final_temp = waypoint_samples[index.item(), :, -1:].cpu()

				path_temp_new = waypoint_samples[:, :, -1:].squeeze().unsqueeze(0).cpu().permute(1, 0, 2)

				if i == 0:
					final = final_temp / resize
					path = path_temp_new / resize
				else:
					final = torch.cat((final, final_temp / resize))
					path = torch.cat((path, path_temp_new / resize), dim=1)
			
			final_list.append(final.squeeze().numpy())
			path_list.append(path.numpy())
			scene_list.append(scene)


		val_ADE = torch.cat(val_ADE).mean()
		val_FDE = torch.cat(val_FDE).mean()

	return final_list, path_list, scene_list, val_ADE.item(), val_FDE.item()
