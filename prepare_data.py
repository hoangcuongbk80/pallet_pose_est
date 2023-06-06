import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
from tqdm import tqdm
from PIL import Image

from handnetAPI import handNet, handGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_hand_points

#root = "/media/hoang/HD-PZFU3/datasets/handnet"
root = "/handnet"

display = False

num_points = 50000
num_hand = 10

remove_outlier = True
valid_obj_idxs = []
hand_labels = {}
camera = 'kinect'
collision_labels = {}

sceneIds = list( range(20, 40)) #100
sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in sceneIds]
        
colorpath = []
depthpath = []
labelpath = []
metapath = []
scenename = []
frameid = []
geometries = []

for x in tqdm(sceneIds, desc = 'Loading data path and collision labels...'):
    for img_num in range(256): #256
        colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
        depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
        labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
        metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
        scenename.append(x.strip())
        frameid.append(img_num)
    
        collision_labels_f = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
        collision_labels[x.strip()] = {}
        for i in range(len(collision_labels_f)):
            collision_labels[x.strip()][i] = collision_labels_f['arr_{}'.format(i)]

def load_hand_labels(root):
    obj_names = list(range(0,88)) #88
    valid_obj_idxs = []
    hand_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading handing labels...')):
        if obj_name == 18: continue
        valid_obj_idxs.append(obj_name) #here align with label png
        label = np.load(os.path.join(root, 'hand_label', '{}_labels.npz'.format(str(obj_name).zfill(3))))
        hand_labels[obj_name] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32))

    return valid_obj_idxs, hand_labels

def get_pointcloud(index):
    color = np.array(Image.open(colorpath[index]), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depthpath[index]))
    seg = np.array(Image.open(labelpath[index]))
    meta = scio.loadmat(metapath[index])
    scene = scenename[index]
    try:
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        poses = meta['poses']
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
        print(scene)
    camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    seg_mask = (seg > 0)
    if remove_outlier:
            camera_poses = np.load(os.path.join(root, 'scenes', scene, camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(root, 'scenes', scene, camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
    else:
        mask = depth_mask

    cloud_masked = cloud[mask]
    color_masked = color[mask]
    seg_masked = seg[mask]

    # sample points
    if len(cloud_masked) >= num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_points-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    seg_sampled = seg_masked[idxs]
    #objectness_label = seg_sampled.copy()
    #objectness_label[objectness_label>1] = 1
    return cloud_sampled, color_sampled, seg_sampled


def get_hand_label(index):
    handnet_root = root # ROOT PATH FOR handNET
    g = handNet(handnet_root, camera='kinect', split='train')

    sceneId = int(scenename[index][-4:])
    annId = frameid[index]
    coef_fric_thresh = 0.2
    camera = 'kinect'

    scenehand = g.loadhandv2(sceneId = sceneId, annId = annId, camera = camera, valid_obj_idxs = valid_obj_idxs, \
                hand_labels = hand_labels, collision_labels = collision_labels, fric_coef_thresh = coef_fric_thresh)

    return scenehand

def compute_votes(scenehand, cloud_sampled, color_sampled, seg_sampled):
    obj_ids = np.unique(seg_sampled)
    hand_group_array = np.zeros((0, 22), dtype=np.float64)

    hand_list = []
    N = cloud_sampled.shape[0]
    point_votes = np.zeros((N,31)) # 10 votes and 1 vote mask 10*3+1 
    point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
    indices = np.arange(N)

    for id in obj_ids:
        obj_idx = id-1 #here align label png with object ids
        if obj_idx not in valid_obj_idxs:
                continue
        hand_mask = (scenehand.hand_group_array[:,16] == obj_idx)
        objecthand = handGroup(scenehand.hand_group_array[hand_mask])

        trans_thresh = 0.1
        rot_thresh = 45.0
        nms_hand = objecthand.nms(translation_thresh = trans_thresh, rotation_thresh = rot_thresh / 180.0 * np.pi)
        while len(nms_hand.hand_group_array) < num_hand and trans_thresh > 0 and rot_thresh > 0:
            trans_thresh = trans_thresh - 0.2
            rot_thresh = rot_thresh - 10.0
            nms_hand = objecthand.nms(translation_thresh = trans_thresh, rotation_thresh = rot_thresh / 180.0 * np.pi)
        if len(nms_hand.hand_group_array) < num_hand:
            continue
        score_sorted_hands = nms_hand.hand_group_array[nms_hand.hand_group_array[:, 0].argsort()]
        objecthand.hand_group_array = score_sorted_hands[::-1] #Reverse the sorted array
        hands = objecthand.hand_group_array[:10]
        hand_group_array = np.concatenate((hand_group_array, hands))

        # Add hand
        for grp in hands:
            hand = np.zeros((8))
            hand[0:3] = np.array([grp[13], grp[14], grp[15]]) # hand_position
            hand[3] = grp[21] # viewpoint
            hand[4] = grp[17] # angle
            hand[5] = grp[0] # quality
            hand[5] = grp[1] # width
            hand[7] = id # semantic class id
            hand_list.append(hand)

        inds = seg_sampled==id
        object_pc = cloud_sampled[inds]
        if len(object_pc) < 200:
            continue

        # Assign first dimension to indicate it belongs an object
        point_votes[inds,0] = 1
        for hand_idx, grp in enumerate(hands):
            hand_position = np.array([grp[0], grp[1], grp[2]])
            # Add the votes (all 0 if the point is not in any object's OBB)
            votes = np.expand_dims(hand_position,0) - object_pc[:,0:3]
            sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
            for i in range(len(sparse_inds)):
                j = sparse_inds[i]
                point_votes[j, int(hand_idx*3+1):int((hand_idx+1)*3+1)] = votes[i,:]

    if len(hand_list)==0:
        final_hands = np.zeros((0,8))
    else:
        final_hands = np.vstack(hand_list) # (K,8)

    scenehand.hand_group_array = hand_group_array

    return scenehand, point_votes, final_hands

def extract_data(data_dir, idx_filename, output_folder):
    
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    exist_idx_list = []
    for exist_file in os.listdir(output_folder):
        if exist_file.endswith("_hand.npy"):
            now_id = int(exist_file[-16:-10])
            exist_idx_list.append(now_id)
    print("Exist files: ", len(exist_idx_list))

    count = len(exist_idx_list)
    for data_idx in data_idx_list:
        print("---------now---------: ", count)

        sceneId = int(scenename[data_idx][-4:])
        annId = frameid[data_idx]
        save_id = sceneId*256 + annId
        if save_id in exist_idx_list:
            print("The id already exist: ", save_id)
            continue
        
        cloud_sampled, color_sampled, seg_sampled = get_pointcloud(data_idx)        
        scenehand = get_hand_label(data_idx)
        scenehand, point_votes, hands = compute_votes(scenehand, cloud_sampled, color_sampled, seg_sampled)
        

        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(save_id)), pc=cloud_sampled)
        np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz'%(save_id)), point_votes = point_votes)
        np.save(os.path.join(output_folder, '%06d_hand.npy'%(save_id)), hands)
        
        if display:
            geometries = []
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_sampled)
            cloud.colors = o3d.utility.Vector3dVector(color_sampled)
            geometries.append(cloud)
            geometries += scenehand.to_open3d_geometry_list()
            o3d.visualization.draw_geometries(geometries)

        count = count+1

if __name__=='__main__':
    idxs = np.array(range(0,len(depthpath)))
    np.random.seed(0)
    np.random.shuffle(idxs)
    
    DATA_DIR = os.path.join(root, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    
    np.savetxt(os.path.join(root, 'data', 'train_data_idx.txt'), idxs[:], fmt='%i')
    
    valid_obj_idxs, hand_labels = load_hand_labels(root)

    extract_data(DATA_DIR, os.path.join(DATA_DIR, 'train_data_idx.txt'), output_folder = os.path.join(DATA_DIR, 'train'))