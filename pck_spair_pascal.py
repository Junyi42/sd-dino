import os
import sys
import torch
torch.set_num_threads(16)
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import json
from glob import glob
from utils.utils_correspondence import pairwise_sim, draw_correspondences_gathered, chunk_cosine_sim, co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace, draw_correspondences_lines
import matplotlib.pyplot as plt
import sys
import time
from utils.logger import get_logger
from loguru import logger
import argparse
from extractor_dino import ViTExtractor
from extractor_sd import load_model, process_features_and_mask, get_mask

def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    if not COUNT_INVIS:
        kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale

def load_spair_data(path, size=256, category='cat', split='test', subsample=None):
    np.random.seed(SEED)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    logger.info(f'Number of SPairs for {category} = {len(pairs)}')
    files = []
    thresholds = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    logger.info(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []
    blank_kps = torch.zeros(num_kps, 3)
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        # thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        # thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]))

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, thresholds

def load_pascal_data(path, size=256, category='cat', split='test', subsample=None):
    
    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    np.random.seed(SEED)
    files = []
    kps = []
    test_data = pd.read_csv(f'{path}/{split}_pairs_pf_pascal.csv')
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    logger.info(f'Number of SPairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    # print(src_img_names.shape, trg_img_names.shape)
    point_A_coords = subset_pairs.iloc[:,3:5]
    point_B_coords = subset_pairs.iloc[:,5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        point_coords_src = get_points(point_A_coords, i).transpose(1,0)
        point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size
        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)
    
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None

def compute_pck(model, aug, save_path, files, kps, category, mask=False, dist='cos', thresholds=None, real_size=960):
    
    img_size = 840 if DINOV2 else 224 if ONLY_DINO else 480
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    stride = 14 if DINOV2 else 4 if ONLY_DINO else 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    
    input_text = "a photo of "+category if TEXT_INPUT else None

    current_save_results = 0
    gt_correspondences = []
    pred_correspondences = []
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2
    pbar = tqdm(total=N)

    for pair_idx in range(N):
        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1_kps = kps[2*pair_idx]

        # Get patch index for the keypoints
        img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()
        img1_y_patch = (num_patches / img_size * img1_y).astype(np.int32)
        img1_x_patch = (num_patches / img_size * img1_x).astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch

        # Load image 2
        img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2_kps = kps[2*pair_idx+1]

        # Get patch index for the keypoints
        img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()
        img2_y_patch = (num_patches / img_size * img2_y).astype(np.int32)
        img2_x_patch = (num_patches / img_size * img2_x).astype(np.int32)
        img2_patch_idx = num_patches * img2_y_patch + img2_x_patch


        with torch.no_grad():
            if not CO_PCA:
                if not ONLY_DINO:
                    img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                    img2_desc = process_features_and_mask(model, aug, img2_input, category, input_text=input_text,  mask=mask).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

            else:
                if not ONLY_DINO:
                    features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text,  mask=False, raw=True)
                    features2 = process_features_and_mask(model, aug, img2_input, input_text=input_text,  mask=False, raw=True)
                    if not RAW:
                        processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
                    else:
                        if WEIGHT[0]:
                            processed_features1 = features1['s5']
                            processed_features2 = features2['s5']
                        elif WEIGHT[1]:
                            processed_features1 = features1['s4']
                            processed_features2 = features2['s4']
                        elif WEIGHT[2]:
                            processed_features1 = features1['s3']
                            processed_features2 = features2['s3']
                        elif WEIGHT[3]:
                            processed_features1 = features1['s2']
                            processed_features2 = features2['s2']
                        else:
                            raise NotImplementedError
                        # rescale the features
                        processed_features1 = F.interpolate(processed_features1, size=(num_patches, num_patches), mode='bilinear', align_corners=False)
                        processed_features2 = F.interpolate(processed_features2, size=(num_patches, num_patches), mode='bilinear', align_corners=False)

                    img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                    img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)
            
            if CO_PCA_DINO:
                cat_desc_dino = torch.cat((img1_desc_dino, img2_desc_dino), dim=2).squeeze() # (1, 1, num_patches**2, dim)
                mean = torch.mean(cat_desc_dino, dim=0, keepdim=True)
                centered_features = cat_desc_dino - mean
                U, S, V = torch.pca_lowrank(centered_features, q=CO_PCA_DINO)
                reduced_features = torch.matmul(centered_features, V[:, :CO_PCA_DINO]) # (t_x+t_y)x(d)
                processed_co_features = reduced_features.unsqueeze(0).unsqueeze(0)
                img1_desc_dino = processed_co_features[:, :, :img1_desc_dino.shape[2], :]
                img2_desc_dino = processed_co_features[:, :, img1_desc_dino.shape[2]:, :]

            if not ONLY_DINO and not RAW: # reweight different layers of sd

                img1_desc[...,:PCA_DIMS[0]]*=WEIGHT[0]
                img1_desc[...,PCA_DIMS[0]:PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[1]
                img1_desc[...,PCA_DIMS[1]+PCA_DIMS[0]:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[2]

                img2_desc[...,:PCA_DIMS[0]]*=WEIGHT[0]
                img2_desc[...,PCA_DIMS[0]:PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[1]
                img2_desc[...,PCA_DIMS[1]+PCA_DIMS[0]:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[2]

            if 'l1' in dist or 'l2' in dist or dist == 'plus_norm':
                # normalize the features
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            if FUSE_DINO and not ONLY_DINO and dist!='plus' and dist!='plus_norm':
                # cat two features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)
                if not RAW:
                    # reweight sd and dino
                    img1_desc[...,:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[3]
                    img1_desc[...,PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]:]*=WEIGHT[4]
                    img2_desc[...,:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[3]
                    img2_desc[...,PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]:]*=WEIGHT[4]

            elif dist=='plus' or dist=='plus_norm':
                img1_desc = img1_desc + img1_desc_dino
                img2_desc = img2_desc + img2_desc_dino
                dist='cos'
            
            if ONLY_DINO:
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino
            # logger.info(img1_desc.shape, img2_desc.shape)

            if DRAW_DENSE:
                mask1 = get_mask(model, aug, img1, category)
                mask2 = get_mask(model, aug, img2, category)

                if ONLY_DINO or not FUSE_DINO:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                
                img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
                img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
                trg_dense_output, src_color_map = find_nearest_patchs(mask2, mask1, img2, img1, img2_desc_reshaped, img1_desc_reshaped, mask=mask, resolution=128)
                if current_save_results!=TOTAL_SAVE_RESULT:
                    if not os.path.exists(f'{save_path}/{category}'):
                        os.makedirs(f'{save_path}/{category}')
                    fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    ax1.axis('off')
                    ax2.axis('off')
                    ax1.imshow(src_color_map)
                    ax2.imshow(trg_dense_output)
                    fig_colormap.savefig(f'{save_path}/{category}/{pair_idx}_colormap.png')
                    plt.close(fig_colormap)
            
            if DRAW_SWAP:
                if not DRAW_DENSE:
                    mask1 = get_mask(model, aug, img1, category)
                    mask2 = get_mask(model, aug, img2, category)

                if ONLY_DINO or not FUSE_DINO:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                    
                img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
                img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
                trg_dense_output, src_color_map = find_nearest_patchs_replace(mask2, mask1, img2, img1, img2_desc_reshaped, img1_desc_reshaped, mask=mask, resolution=156, draw_gif=DRAW_GIF, save_path=f'{save_path}/{category}/{pair_idx}_swap.gif')
                if current_save_results!=TOTAL_SAVE_RESULT:
                    if not os.path.exists(f'{save_path}/{category}'):
                        os.makedirs(f'{save_path}/{category}')
                    fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    ax1.axis('off')
                    ax2.axis('off')
                    ax1.imshow(src_color_map)
                    ax2.imshow(trg_dense_output)
                    fig_colormap.savefig(f'{save_path}/{category}/{pair_idx}_swap.png')
                    plt.close(fig_colormap)

        if MASK and CO_PCA:
            mask2 = get_mask(model, aug, img2, category)
            img2_desc = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
            resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest')
            img2_desc = img2_desc * resized_mask2.repeat(1, img2_desc.shape[1], 1, 1)
            img2_desc[(img2_desc.sum(dim=1)==0).repeat(1, img2_desc.shape[1], 1, 1)] = 100000
            # reshape back
            img2_desc = img2_desc.reshape(1, 1, img2_desc.shape[1], num_patches*num_patches).permute(0,1,3,2)

        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        if COUNT_INVIS:
            vis = torch.ones_like(vis)
        # Get similarity matrix
        if dist == 'cos':
            sim_1_to_2 = chunk_cosine_sim(img1_desc, img2_desc).squeeze()
        elif dist == 'l2':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2).squeeze()
        elif dist == 'l1':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1).squeeze()
        elif dist == 'l2_norm':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2, normalize=True).squeeze()
        elif dist == 'l1_norm':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1, normalize=True).squeeze()
        else:
            raise ValueError('Unknown distance metric')

        # Get nearest neighors
        nn_1_to_2 = torch.argmax(sim_1_to_2[img1_patch_idx], dim=1)
        nn_y_patch, nn_x_patch = nn_1_to_2 // num_patches, nn_1_to_2 % num_patches
        nn_x = (nn_x_patch - 1) * stride + stride + patch_size // 2 - .5
        nn_y = (nn_y_patch - 1) * stride + stride + patch_size // 2 - .5
        kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)

        gt_correspondences.append(img2_kps[vis][:, [1,0]])
        pred_correspondences.append(kps_1_to_2[vis][:, [1,0]])
        if thresholds is not None:
            bbox_size.append(thresholds[pair_idx].repeat(vis.sum()))
        
        if current_save_results!=TOTAL_SAVE_RESULT:
            tmp_alpha = torch.tensor([0.1, 0.05, 0.01])
            if thresholds is not None:
                tmp_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
                tmp_threshold = tmp_alpha.unsqueeze(-1) * tmp_bbox_size.unsqueeze(0)
            else:
                tmp_threshold = tmp_alpha * img_size
            if not os.path.exists(f'{save_path}/{category}'):
                os.makedirs(f'{save_path}/{category}')
            # fig=draw_correspondences_gathered(img1_kps[vis][:, [1,0]], kps_1_to_2[vis][:, [1,0]], img1, img2)
            fig=draw_correspondences_lines(img1_kps[vis][:, [1,0]], kps_1_to_2[vis][:, [1,0]], img2_kps[vis][:, [1,0]], img1, img2, tmp_threshold)
            fig.savefig(f'{save_path}/{category}/{pair_idx}_pred.png')
            fig_gt=draw_correspondences_gathered(img1_kps[vis][:, [1,0]], img2_kps[vis][:, [1,0]], img1, img2)
            fig_gt.savefig(f'{save_path}/{category}/{pair_idx}_gt.png')
            plt.close(fig)
            plt.close(fig_gt)
            current_save_results+=1

        pbar.update(1)

    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    alpha = torch.tensor([0.1, 0.05, 0.01]) if not PASCAL else torch.tensor([0.1, 0.05, 0.15])
    correct = torch.zeros(len(alpha))

    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alpha), 1)
    if thresholds is not None:
        bbox_size = torch.cat(bbox_size, dim=0).cpu()
        threshold = alpha.unsqueeze(-1) * bbox_size.unsqueeze(0)
        correct = err < threshold
    else:
        threshold = alpha * img_size
        correct = err < threshold.unsqueeze(-1)

    correct = correct.sum(dim=-1) / len(gt_correspondences)

    alpha2pck = zip(alpha.tolist(), correct.tolist())
    logger.info(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                    for alpha, pck_alpha in alpha2pck]))

    return correct

def main(args):
    global MASK, SAMPLE, DIST, COUNT_INVIS, TOTAL_SAVE_RESULT, BBOX_THRE, VER, CO_PCA, PCA_DIMS, SIZE, FUSE_DINO, DINOV2, MODEL_SIZE, DRAW_DENSE, TEXT_INPUT, DRAW_SWAP, ONLY_DINO, SEED, EDGE_PAD, WEIGHT, CO_PCA_DINO, PASCAL, DRAW_GIF, RAW
    MASK = args.MASK
    SAMPLE = args.SAMPLE
    DIST = args.DIST
    COUNT_INVIS = args.COUNT_INVIS
    TOTAL_SAVE_RESULT = args.TOTAL_SAVE_RESULT
    BBOX_THRE = False if args.IMG_THRESHOLD else True
    VER = args.VER
    CO_PCA = False if args.PROJ_LAYER else True
    CO_PCA_DINO = args.CO_PCA_DINO
    PCA_DIMS = args.PCA_DIMS
    SIZE = args.SIZE
    INDICES = args.INDICES
    EDGE_PAD = args.EDGE_PAD

    FUSE_DINO = False if args.NOT_FUSE else True
    ONLY_DINO = args.ONLY_DINO
    DINOV2 = False if args.DINOV1 else True
    MODEL_SIZE = args.MODEL_SIZE
    
    DRAW_DENSE = args.DRAW_DENSE
    DRAW_SWAP = args.DRAW_SWAP
    DRAW_GIF = args.DRAW_GIF
    TEXT_INPUT = args.TEXT_INPUT
    
    SEED = args.SEED
    WEIGHT = args.WEIGHT # corresponde to three groups for the sd features, and one group for the dino features
    PASCAL = args.PASCAL
    RAW = args.RAW

    if SAMPLE == 0:
        SAMPLE = None
    if DRAW_DENSE or DRAW_SWAP:
        TOTAL_SAVE_RESULT = SAMPLE
        MASK = True
    if ONLY_DINO:
        FUSE_DINO = True
    if FUSE_DINO and not ONLY_DINO:
        DIST = "l2"
    else:
        DIST = "cos"
    if args.DIST != "cos" and args.DIST != "l2":
        DIST = args.DIST
    if PASCAL:
        SAMPLE = 0

    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True
    model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=args.TIMESTEP, block_indices=tuple(INDICES))
    save_path=f'./results_spair/pck_fuse_{args.NOTE}mask_{MASK}_sample_{SAMPLE}_BBOX_{BBOX_THRE}_dist_{DIST}_Invis_{COUNT_INVIS}_{args.TIMESTEP}{VER}_{MODEL_SIZE}_{SIZE}_copca_{CO_PCA}_{INDICES[0]}_{PCA_DIMS[0]}_{INDICES[1]}_{PCA_DIMS[1]}_{INDICES[2]}_{PCA_DIMS[2]}_text_{TEXT_INPUT}_sd_{WEIGHT[3]}{not ONLY_DINO}_dino_{WEIGHT[4]}{FUSE_DINO}'
    if PASCAL:
        save_path=f'./results_pascal/pck_fuse_{args.NOTE}mask_{MASK}_sample_{SAMPLE}_BBOX_{BBOX_THRE}_dist_{DIST}_Invis_{COUNT_INVIS}_{args.TIMESTEP}{VER}_{MODEL_SIZE}_{SIZE}_copca_{CO_PCA}_{INDICES[0]}_{PCA_DIMS[0]}_{INDICES[1]}_{PCA_DIMS[1]}_{INDICES[2]}_{PCA_DIMS[2]}_text_{TEXT_INPUT}_sd_{WEIGHT[3]}{not ONLY_DINO}_dino_{WEIGHT[4]}{FUSE_DINO}'
    if EDGE_PAD:
        save_path += '_edge_pad'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logger = get_logger(save_path+'/result.log')

    logger.info(args)
    data_dir = 'data/SPair-71k' if not PASCAL else 'data/PF-dataset-PASCAL'
    if not PASCAL:
        categories = os.listdir(os.path.join(data_dir, 'ImageAnnotation'))
        categories = sorted(categories)
    else:
        categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] # for pascal
    img_size = 840 if DINOV2 else 224 if ONLY_DINO else 480

    pcks = []
    pcks_05 = []
    pcks_01 = []
    start_time=time.time()
    for cat in categories:
        files, kps, thresholds = load_spair_data(data_dir, size=img_size, category=cat, subsample=SAMPLE) if not PASCAL else load_pascal_data(data_dir, size=img_size, category=cat, subsample=SAMPLE)
        if BBOX_THRE:
            pck = compute_pck(model, aug, save_path, files, kps, cat, mask=MASK, dist=DIST, thresholds=thresholds, real_size=SIZE)
        else:
            pck = compute_pck(model, aug, save_path, files, kps, cat, mask=MASK, dist=DIST, real_size=SIZE)
        pcks.append(pck[0])
        pcks_05.append(pck[1])
        pcks_01.append(pck[2])
    end_time=time.time()
    minutes, seconds = divmod(end_time-start_time, 60)
    logger.info(f"Time: {minutes:.0f}m {seconds:.0f}s")
    logger.info(f"Average PCK0.10: {np.average(pcks) * 100:.2f}")
    logger.info(f"Average PCK0.05: {np.average(pcks_05) * 100:.2f}")
    logger.info(f"Average PCK0.01: {np.average(pcks_01) * 100:.2f}") if not PASCAL else logger.info(f"Average PCK0.15: {np.average(pcks_01) * 100:.2f}")
    if SAMPLE is None or SAMPLE==0:
        weights_pascal=[15,30,10,6,8,32,19,27,13,3,8,24,9,27,12,7,1,13,20,15]
        weights_spair=[690,650,702,702,870,644,564,600,646,640,600,600,702,650,862,664,756,692]
        weights = weights_pascal if PASCAL else weights_spair
    else:
        weights = [1] * len(pcks)
    logger.info(f"Weighted PCK0.10: {np.average(pcks, weights=weights) * 100:.2f}")
    logger.info(f"Weighted PCK0.05: {np.average(pcks_05, weights=weights) * 100:.2f}")
    logger.info(f"Weighted PCK0.01: {np.average(pcks_01, weights=weights) * 100:.2f}") if not PASCAL else logger.info(f"Weighted PCK0.15: {np.average(pcks_01, weights=weights) * 100:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--MASK', action='store_true', default=False)               # set true to use the segmentation mask for the extracted features
    parser.add_argument('--SAMPLE', type=int, default=20)                           # sample 20 pairs for each category, set to 0 to use all pairs
    parser.add_argument('--DIST', type=str, default='l2')                           # distance metric, cos, l2, l1, l2_norm, l1_norm, plus, plus_norm
    parser.add_argument('--COUNT_INVIS', action='store_true', default=False)        # set true to count invisible keypoints
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=5)                 # save the qualitative results for the first 5 pairs
    parser.add_argument('--IMG_THRESHOLD', action='store_true', default=False)      # set the pck threshold to the image size rather than the bbox size
    parser.add_argument('--VER', type=str, default="v1-5")                          # version of diffusion, v1-3, v1-4, v1-5, v2-1-base
    parser.add_argument('--PROJ_LAYER', action='store_true', default=False)         # set true to use the pretrained projection layer from ODISE for dimension reduction
    parser.add_argument('--CO_PCA_DINO', type=int, default=0)                       # whether perform co-pca on dino features
    parser.add_argument('--PCA_DIMS', nargs=3, type=int, default=[256, 256, 256])   # the dimensions of the three groups of sd features
    parser.add_argument('--TIMESTEP', type=int, default=100)                        # timestep for diffusion, [0, 1000], 0 for no noise added
    parser.add_argument('--SIZE', type=int, default=960)                            # image size for the sd input
    parser.add_argument('--INDICES', nargs=4, type=int, default=[2,5,8,11])         # select different layers of sd features, only the first three are used by default
    parser.add_argument('--EDGE_PAD', action='store_true', default=False)           # set true to pad the image with the edge pixels
    parser.add_argument('--WEIGHT', nargs=5, type=float, default=[1,1,1,1,1])       # first three corresponde to three layers for the sd features, and the last two for the ensembled sd/dino features
    parser.add_argument('--RAW', action='store_true', default=False)                # set true to use the raw features from sd

    parser.add_argument('--NOT_FUSE', action='store_true', default=False)           # set true to use only sd features
    parser.add_argument('--ONLY_DINO', action='store_true', default=False)          # set true to use only dino features
    parser.add_argument('--DINOV1',  action='store_true', default=False)            # set true to use dinov1
    parser.add_argument('--MODEL_SIZE', type=str, default='base')                   # model size of thye dinov2, small, base, large

    parser.add_argument('--DRAW_DENSE', action='store_true', default=False)         # set true to draw the dense correspondences
    parser.add_argument('--DRAW_SWAP', action='store_true', default=False)          # set true to draw the swapped images
    parser.add_argument('--DRAW_GIF', action='store_true', default=False)           # set true to generate the gif for the swapped images
    parser.add_argument('--TEXT_INPUT', action='store_true', default=False)         # set true to use the explicit text input

    parser.add_argument('--PASCAL', action='store_true', default=False)             # set true to test on pfpascal dataset
    parser.add_argument('--NOTE', type=str, default='')

    args = parser.parse_args()
    main(args)