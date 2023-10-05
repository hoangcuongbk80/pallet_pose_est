"""
conda activate BOP_toolkit_env
pip install .
python scripts/vis_gt_poses_ycbv_cuong.py
"""

import os
import numpy as np
import cv2

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visualization_cuong


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'ycbv',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # File with a list of estimation targets used to determine the set of images
  # for which the GT poses will be visualized. The file is assumed to be stored
  # in the dataset folder. None = all images.
  # 'targets_filename': 'test_targets_bop19.json',
  'targets_filename': None,

  # Select ID's of scenes, images and GT poses to be processed.
  # Empty list [] means that all ID's will be used.
  'scene_ids': [],
  'im_ids': [],
  'gt_ids': [],

  # Indicates whether to render RGB images.
  'vis_rgb': True,

  # Indicates whether to resolve visibility in the rendered RGB images (using
  # depth renderings). If True, only the part of object surface, which is not
  # occluded by any other modeled object, is visible. If False, RGB renderings
  # of individual objects are blended together.
  'vis_rgb_resolve_visib': True,

  # Indicates whether to save images of depth differences.
  'vis_depth_diff': True,

  # Whether to use the original model color.
  'vis_orig_color': False,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # Folder for output visualisations.
  'vis_path': os.path.join(config.output_path, 'vis_gt_poses_cuong'),

  # Path templates for output images.
  'vis_rgb_tpath': os.path.join(
    '{vis_path}', '{dataset}', '{split}', '{scene_id:06d}', '{im_id:06d}.jpg'),
  'vis_depth_diff_tpath': os.path.join(
    '{vis_path}', '{dataset}', '{split}', '{scene_id:06d}',
    '{im_id:06d}_depth_diff.jpg'),
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = 'eval'  # None = default.
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

# Load colors.
colors_path = os.path.join(
  os.path.dirname(visualization_cuong.__file__), 'colors.json')
colors = inout.load_json(colors_path)

# Subset of images for which the ground-truth poses will be rendered.
if p['targets_filename'] is not None:
  targets = inout.load_json(
    os.path.join(dp_split['base_path'], p['targets_filename']))
  scene_im_ids = {}
  for target in targets:
    scene_im_ids.setdefault(
      target['scene_id'], set()).add(target['im_id'])
else:
  scene_im_ids = None

# List of considered scenes.
scene_ids_curr = dp_split['scene_ids']
if p['scene_ids']:
  scene_ids_curr = set(scene_ids_curr).intersection(p['scene_ids'])

# Rendering mode.
renderer_modalities = []
if p['vis_rgb']:
  renderer_modalities.append('rgb')
if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
  renderer_modalities.append('depth')
renderer_mode = '+'.join(renderer_modalities)

# Create a renderer.
width, height = dp_split['im_size']
ren = renderer.create_renderer(
  width, height, p['renderer_type'], mode=renderer_mode, shading='flat')

# Load object models.
models = {}
for obj_id in dp_model['obj_ids']:
  misc.log('Loading 3D model of object {}...'.format(obj_id))
  model_path = dp_model['model_tpath'].format(obj_id=obj_id)
  model_color = None
  if not p['vis_orig_color']:
    model_color = tuple(colors[(obj_id - 1) % len(colors)])
  ren.add_object(obj_id, model_path, surf_color=model_color)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
for scene_id in scene_ids:

  # Load scene info and ground-truth poses.
  scene_camera = inout.load_scene_camera(
    dp_split['scene_camera_tpath'].format(scene_id=scene_id))
  scene_gt = inout.load_scene_gt(
    dp_split['scene_gt_tpath'].format(scene_id=scene_id))

  # List of considered images.
  if scene_im_ids is not None:
    im_ids = scene_im_ids[scene_id]
  else:
    im_ids = sorted(scene_gt.keys())
  if p['im_ids']:
    im_ids = set(im_ids).intersection(p['im_ids'])

  # Render the object models in the ground-truth poses in the selected images.
  for im_counter, im_id in enumerate(im_ids):
    if im_counter % 10 == 0:
      misc.log(
        'Visualizing GT poses - dataset: {}, scene: {}, im: {}/{}'.format(
          p['dataset'], scene_id, im_counter, len(im_ids)))

    K = scene_camera[im_id]['cam_K']

    # List of considered ground-truth poses.
    gt_ids_curr = range(len(scene_gt[im_id]))
    if p['gt_ids']:
      gt_ids_curr = set(gt_ids_curr).intersection(p['gt_ids'])

    # Collect the ground-truth poses.
    gt_poses = []
    for gt_id in gt_ids_curr:
      gt = scene_gt[im_id][gt_id]
      gt_poses.append({
        'obj_id': gt['obj_id'],
        'R': gt['cam_R_m2c'],
        't': gt['cam_t_m2c'],
        'text_info': [
          {'name': '', 'val': '{}:{}'.format(gt['obj_id'], gt_id), 'fmt': ''}
        ]
      })

    # Load the color and depth images and prepare images for rendering.
    rgb = None
    if p['vis_rgb']:
      if 'rgb' in dp_split['im_modalities'] or p['dataset_split_type'] == 'pbr':
        rgb = inout.load_im(dp_split['rgb_tpath'].format(
          scene_id=scene_id, im_id=im_id))[:, :, :3]
      elif 'gray' in dp_split['im_modalities']:
        gray = inout.load_im(dp_split['gray_tpath'].format(
          scene_id=scene_id, im_id=im_id))
        rgb = np.dstack([gray, gray, gray])
      else:
        raise ValueError('RGB nor gray images are available.')

    rgb_file = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)
    #print("Cuong: ", rgb_file)

    # Path to the output RGB visualization_cuong.
    vis_rgb_path = None
    if p['vis_rgb']:
      vis_rgb_path = p['vis_rgb_tpath'].format(
        vis_path=p['vis_path'], dataset=p['dataset'], split=p['dataset_split'],
        scene_id=scene_id, im_id=im_id)



    # visualization_cuong.
    # intrinsics of YCB-VIDEO dataset
    k_ycbvideo = np.array([[1066.778, 0, 312.9869],
                            [0, 1067.487, 241.3109],
                            [0, 0, 1]])
    # 21 objects for YCB-Video dataset
    object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                              '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                              '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                              '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
    vertex_ycbvideo = np.load('./data/YCB-Video/YCB_vertex.npy')
    visualization_cuong.vis_object_poses(
      poses=gt_poses, k_ycbvideo=k_ycbvideo, object_names_ycbvideo=object_names_ycbvideo, vertex_ycbvideo=vertex_ycbvideo,
      K=K, renderer=ren, rgb=rgb, rgb_file=rgb_file, vis_rgb_path=vis_rgb_path)
    #if im_counter % 10 != 0:
    break
  break

misc.log('Done.')
