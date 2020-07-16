import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
from manopth import demo

import numpy as np
import json
import os
import trimesh
import argparse

from sklearn.neighbors import KDTree

class pcl2mano():
    def __init__(self, device, mano_root = 'mano/models', ncomps = 45, seg_file= './face2label_sealed.npy', visualize_interm_result = False):
        self.ncomps = ncomps
        self.device = device
        self.mano_root = mano_root
        self.mano_layer = ManoLayer(mano_root=self.mano_root, use_pca=True, ncomps=self.ncomps, flat_hand_mean=False)
        self.mano_layer = self.mano_layer.to(self.device)

        segmentation = np.load(seg_file)
        self.faces = self.mano_layer.th_faces.detach().cpu()
        # assign mano vertex label according to face label:
        self.vertex_label = np.zeros((self.faces.max()+1), dtype = np.uint8)

        for i in range(0,self.faces.shape[0]):
            self.vertex_label[self.faces[i,0]] = segmentation[i,0]
            self.vertex_label[self.faces[i,1]] = segmentation[i,0]
            self.vertex_label[self.faces[i,2]] = segmentation[i,0]

        self.label_vertex = [(np.where(self.vertex_label == 0))[0], \
                            (np.where(self.vertex_label == 1))[0], \
                            (np.where(self.vertex_label == 2))[0], \
                            (np.where(self.vertex_label == 3))[0], \
                            (np.where(self.vertex_label == 4))[0], \
                            (np.where(self.vertex_label == 5))[0]]

        self.visualize_interm_result = visualize_interm_result

        self.target_points = []
        self.target_points_tree = []
        self.no_label = []

    def find_nearest_neighbour_index_from_hand(self, target_points_tree, hands, no_label):
        if len(hands.shape) == 3: # batch of hands
            closest_indices = []
            for i in range(0, hands.shape[0]):
                hand = hands[i,:,:].detach().cpu().numpy()
                _, closest_palm_index = target_points_tree[0].query(hand[self.label_vertex[0], :], 1)
                _, closest_thum_index = target_points_tree[1].query(hand[self.label_vertex[1], :], 1)
                _, closest_inde_index = target_points_tree[2].query(hand[self.label_vertex[2], :], 1)
                _, closest_midd_index = target_points_tree[3].query(hand[self.label_vertex[3], :], 1)
                _, closest_ring_index = target_points_tree[4].query(hand[self.label_vertex[4], :], 1)
                _, closest_pink_index = target_points_tree[5].query(hand[self.label_vertex[5], :], 1)

                closest_indices.append([closest_palm_index, closest_thum_index, closest_inde_index,\
                    closest_midd_index, closest_ring_index, closest_pink_index])

            return closest_indices
        else: # Not used
            print("Fatal error in find_nearest_neighbour_index_from_hand!")
            exit()

    def index2points_from_hand(self,indices, closest_points):
        for batch in range(0, len(indices)): # iterate over batch dimension
            for label in range(0, 6): # iterate over hand parts
                closest_points[batch, self.label_vertex[label][:], :] = self.target_points[label][indices[batch][label][:,0],:]

        return closest_points

    def find_nearest_neighbour_to_hand(self, target_points, n, hands, no_label):
        closest_batch_hand_indices = np.zeros((0, n), dtype = np.int32)
        target_points_batch_rearrange = np.zeros((0, n, 3), dtype = np.float32)
        for i in range(0, hands.shape[0]): # iterate over batch dimension
            hand = hands[i,:,:].detach().cpu().numpy()
            closest_hand_indices = np.zeros((0), dtype=np.int32)
            target_points_rearrange = np.zeros((0,3), dtype=np.float32)
            for hand_part in range(0,6):
                # If there is no predicted label of that part, skip.
                if no_label[hand_part]:
                    continue
                hand_part_vertex = hand[self.label_vertex[hand_part],:]
                hand_part_tree = KDTree(hand_part_vertex)
                _, closest_part_index = hand_part_tree.query(target_points[hand_part],1)
                closest_hand_indices = np.concatenate((closest_hand_indices, self.label_vertex[hand_part][closest_part_index[:,0]]), 0)
                target_points_rearrange = np.concatenate((target_points_rearrange, target_points[hand_part]))
            target_points_rearrange = np.expand_dims(target_points_rearrange, axis = 0)

            target_points_batch_rearrange = np.concatenate((target_points_batch_rearrange, target_points_rearrange), 0)
            closest_hand_indices = np.expand_dims(closest_hand_indices, axis = 0)
            closest_batch_hand_indices = np.concatenate((closest_batch_hand_indices, closest_hand_indices),0)

        return closest_batch_hand_indices, target_points_batch_rearrange

    def index2points_to_hand(self, indices, hands):
        hands_batch_rearrange = torch.zeros(0, indices.shape[1], 3).float().to(self.device)
        for batch in range(0, indices.shape[0]):
            hand_rearrange = hands[batch, indices[batch, :], :]
            hand_rearrange = hand_rearrange.unsqueeze(0)
            hands_batch_rearrange = torch.cat((hands_batch_rearrange, hand_rearrange), dim=0)# axis = 0)
        return hands_batch_rearrange

    
    def mask_no_label_from_hand(self, hand_verts, no_label):
        for i in range(len(no_label)):
            if no_label[i]:
                hand_verts[:, self.label_vertex[i], :] = 0.
        return hand_verts
    

    def mask_no_label_to_hand(self, hand_verts, no_label):
        for i in range(len(no_label)):
            if no_label[i]:
                hand_verts[:, self.label_vertex[i], :] = 0.
        return hand_verts
    

    def fit_mano_2_pcl(self, samples, # samples is a (N_v x 3) array, unit is Millimeter!
                        labels, # labels is a (N_v x 1) array
                        seeds = 8, coarse_iter = 50, fine_iter = 50, stop_loss = 5.0, verbose=0):
        # classify samples according to labels
        palm = samples[(np.where(labels == 0))[0], :]
        thumb = samples[(np.where(labels == 1))[0], :]
        index = samples[(np.where(labels == 2))[0], :]
        middle = samples[(np.where(labels == 3))[0], :]
        ring = samples[(np.where(labels == 4))[0], :]
        pinky = samples[(np.where(labels == 5))[0], :]
        for lab in [palm, thumb, index, middle, ring, pinky]:
            # print(len(lab))
            # print(lab.shape)
            self.no_label.append(lab.shape[0] == 0)
        # Add temp point (0,0,0) to part with no sample.
        # The loss from these points will be masked out later when calculating loss
        if palm.shape[0] == 0: palm = np.zeros([1,3])
        if thumb.shape[0] == 0: thumb = np.zeros([1,3])
        if index.shape[0] == 0: index = np.zeros([1,3])
        if middle.shape[0] == 0: middle = np.zeros([1,3])
        if ring.shape[0] == 0: ring = np.zeros([1,3])
        if pinky.shape[0] == 0: pinky = np.zeros([1,3])

        # print("No label:", self.no_label)
        self.target_points_np = [palm, thumb, index, middle, ring, pinky]
        self.target_points = [torch.from_numpy(palm).float().to(self.device),\
                            torch.from_numpy(thumb).float().to(self.device),\
                            torch.from_numpy(index).float().to(self.device),\
                            torch.from_numpy(middle).float().to(self.device),\
                            torch.from_numpy(ring).float().to(self.device),\
                            torch.from_numpy(pinky).float().to(self.device)]
        self.target_points_tree = [KDTree(palm), KDTree(thumb), KDTree(index), KDTree(middle), KDTree(ring), KDTree(pinky)]
        
        # Model para initialization:
        shape = torch.zeros(seeds, 10).float().to(self.device)
        shape.requires_grad_()
        rot = torch.zeros(seeds, 3).float().to(self.device)
        rot.requires_grad_()
        pose = torch.zeros(seeds, self.ncomps).float().to(self.device)
        pose = (0.1*torch.randn(seeds, self.ncomps)).float().to(self.device)
        pose.requires_grad_()
        trans = torch.from_numpy(samples.mean(0)/1000.0) # trans should be in meter
        trans = trans.unsqueeze(0).repeat(seeds, 1).float().to(self.device)
        # trans = (0.1*torch.randn(seeds, 3)).float().to(self.device)
        trans.requires_grad_()

        hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose),1), shape, trans)

        if self.visualize_interm_result:
            demo.display_mosh(torch.from_numpy(samples).float().unsqueeze(0).expand(seeds, -1, -1),\
                                np.zeros((0,4), dtype = np.int32),
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=self.mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.3)

        # Global optimization
        criteria_loss = nn.MSELoss().to(self.device)
        previous_loss = 1e8
        optimizer = torch.optim.Adam([trans, rot], lr=1e-2)
        print('...Optimizing global transformation...')
        for i in range(0, coarse_iter):
            hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose),1), shape, trans)
            # Find closest label points:
            closest_indices = self.find_nearest_neighbour_index_from_hand(self.target_points_tree, hand_verts, self.no_label)
            closest_points = self.index2points_from_hand(closest_indices, torch.zeros_like(hand_verts))

            for j in range(0,20):   
                hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose),1), shape, trans)
                hand_verts = self.mask_no_label_from_hand(hand_verts, self.no_label)
                loss = criteria_loss(hand_verts, closest_points)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = criteria_loss(hand_verts, closest_points)
            if verbose >= 1:
                print(i, loss.data)
            if previous_loss - loss.data < 1e-1:
                break
            previous_loss = loss.data.detach()
        # print('After coarse alignment: %6f'%(loss.data))
        if self.visualize_interm_result:
            demo.display_mosh(torch.from_numpy(samples).float().unsqueeze(0).expand(seeds, -1, -1),\
                                np.zeros((0,4), dtype = np.int32),
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=self.mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.3)

        # Local optimization
        previous_loss = 1e8
        optimizer = torch.optim.Adam([trans, rot, pose, shape], lr=1e-2)
        print('...Optimizing hand pose shape and global transformation...')
        for i in range(0, fine_iter):
            hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose),1), shape, trans)
            # Find closest label points:
            closest_batch_hand_indices, target_points_batch_rearrange = self.find_nearest_neighbour_to_hand(
                                                                            self.target_points_np, samples.shape[0], hand_verts, self.no_label
                                                                        )
            target_points_batch_rearrange = torch.from_numpy(target_points_batch_rearrange).float().to(self.device)

            for j in range(0,20):   
                hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose),1), shape, trans) 
                hands_batch_rearrange = self.index2points_to_hand(closest_batch_hand_indices, hand_verts)

                w_pose = 100.0
                loss = criteria_loss(hands_batch_rearrange, target_points_batch_rearrange) + w_pose*(pose*pose).mean() # pose regularizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = criteria_loss(hands_batch_rearrange, target_points_batch_rearrange)

            # if previous_loss - loss.data < 1e-1:
            #     break
            previous_loss = loss.data.detach()

            # Find smallest loss in the #seeds seeds:
            per_seed_error = ((hands_batch_rearrange - target_points_batch_rearrange)*(hands_batch_rearrange - target_points_batch_rearrange)).mean(2).mean(1)
            min_index = torch.argmin(per_seed_error).detach().cpu().numpy()
            min_error = per_seed_error[min_index]
            if verbose >=1: print(i, min_error.data)

            if self.visualize_interm_result and i%40 == 0:
                tmp_arange = np.expand_dims(np.arange(target_points_batch_rearrange.shape[1]), axis = 1)
                link = np.concatenate((tmp_arange, tmp_arange+target_points_batch_rearrange.shape[1]), 1)
                link = link[0:200,:]
                visual_points = torch.cat((target_points_batch_rearrange[min_index,:,:], hands_batch_rearrange[min_index,:,:]), dim = 0)
                visual_points = visual_points.unsqueeze(0).expand(seeds, -1, -1)

                pass
                demo.display_mosh(visual_points.detach().cpu(),\
                                link,
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=self.mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.3)

            if min_error < stop_loss:
                break

        # print('After fine alignment: %6f'%(loss.data))

        hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose),1), shape, trans)

        hand_shape = {'vertices': hand_verts.detach().cpu().numpy()[min_index, :, :], \
                        'joints': hand_joints.detach().cpu().numpy()[min_index, :, :], \
                         'faces': self.mano_layer.th_faces.detach().cpu()}

        mano_para = {'rot': rot.detach().cpu().numpy()[min_index, :], \
                    'pose': pose.detach().cpu().numpy()[min_index, :], \
                    'shape': shape.detach().cpu().numpy()[min_index, :], \
                    'trans': trans.detach().cpu().numpy()[min_index, :]}

        return hand_shape, mano_para


def fit_mano(points_npz_filename, out_name,return_mano=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solver = pcl2mano(device, mano_root='./mano/models', 
                ncomps=30, seg_file='./pcl2mano/face2label_sealed.npy')
    # samples, labels = create_target_data(args.data_file, args.seg_file)

    data = np.load(points_npz_filename)
    samples = data['points']
    labels = data['labels']
    samples = samples * 1000 # 200
     
    hand_shape, mano_para = solver.fit_mano_2_pcl(samples, labels)

    # convert (mm) back to (m)
    hand_shape['joints'] /= 1000.0
    hand_shape['vertices'] /= 1000.0

    mesh = trimesh.Trimesh(hand_shape['vertices'] , hand_shape["faces"], process=False)
    mesh = seal(mesh)
    print(" -> Save MANO mesh to", out_name)
    mesh.export(out_name)


def create_target_data(hand_mesh_file, seg_file):
    hand_mesh = trimesh.load(hand_mesh_file, process = False)
    hand_mesh.vertices = hand_mesh.vertices*1000.0
    seg = np.load(seg_file)
    samples, faceId = trimesh.sample.sample_surface_even(hand_mesh, 800)
    labels = np.expand_dims(seg[faceId, 0], axis=1)
    
    return samples, labels


def seal(mesh_to_seal):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal
