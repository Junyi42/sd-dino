import os
import sys
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import time
from utils.logger import get_logger
from loguru import logger
import argparse
from extractor_sd import load_model, process_features_and_mask, get_mask
from extractor_dino import ViTExtractor
from utils.utils_tss import TSSDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import imageio
from imageio import imwrite
from utils.utils_flow import remap_using_flow_fields, flow_to_image, convert_flow_to_mapping, overlay_semantic_mask
import torch.nn.functional as F 

def get_smooth(img, mask=None):

    if mask is not None:
        img_smooth=img.clone().permute(0, 2, 3, 1)
        img_smooth[~mask] = 0
        img=img_smooth.permute(0, 3, 1, 2)

    def _gradient_x(img,mask): #tobe implemented
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def _gradient_y(img,mask):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy
        
    img_grad_x = _gradient_x(img, mask)
    img_grad_y = _gradient_y(img, mask)

    if mask is not None:
        smooth = (torch.abs(img_grad_x).sum() + torch.abs(img_grad_y).sum())/torch.sum(mask)
    else:
        smooth = torch.mean(torch.abs(img_grad_x)) + torch.mean(torch.abs(img_grad_y))

    return smooth


def plot_individual_images(save_path, name_image, source_image, target_image, flow_est, flow_gt,
                           mask_used=None, color=[255, 102, 51]):
    if not isinstance(source_image, np.ndarray):
        source_image = source_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        target_image = target_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        # numpy array
        if not source_image.shape[2] == 3:
            source_image = source_image.transpose(1, 2, 0)
            target_image = target_image.transpose(1, 2, 0)

    flow_target = flow_est.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    flow_gt = flow_gt.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    remapped_est = remap_using_flow_fields(source_image, flow_target[:, :, 0], flow_target[:, :, 1])

    max_mapping = 520
    max_flow = 400
    rgb_flow = flow_to_image(flow_target, max_flow)
    rgb_flow_gt = flow_to_image(flow_gt, max_flow)
    rgb_mapping = flow_to_image(convert_flow_to_mapping(flow_target, False), max_mapping)

    if not os.path.isdir(os.path.join(save_path, 'individual_images')):
        os.makedirs(os.path.join(save_path, 'individual_images'))
    # save the rgb flow
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_rgb_flow.png".format(name_image)), rgb_flow)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_rgb_flow_gt.png".format(name_image)), rgb_flow_gt)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_rgb_mapping.png".format(name_image)),rgb_mapping)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_image_s.png".format(name_image)), source_image)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_image_t.png".format(name_image)), target_image)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_warped_s.png".format(name_image)),
                    remapped_est)

    if mask_used is not None:
        mask_used = mask_used.squeeze().cpu().numpy()
        imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_mask.png".format(name_image)),
                        mask_used.astype(np.uint8) * 255)
        imageio.imwrite(
            os.path.join(save_path, 'individual_images', "{}_image_s_warped_and_mask.png".format(name_image)),
            remapped_est * np.tile(np.expand_dims(mask_used.astype(np.uint8), axis=2), (1, 1, 3)))

        # overlay mask on warped image
        img_mask_overlay_color = overlay_semantic_mask(remapped_est.astype(np.uint8),
                                                       255 - mask_used.astype(np.uint8) * 255, color=color)
        imwrite(os.path.join(save_path, 'individual_images',
                             '{}_warped_overlay_mask_color.png'.format(name_image)), img_mask_overlay_color)

        flow_mask_overlay_color = overlay_semantic_mask(rgb_flow, 255 - mask_used.astype(np.uint8) * 255, color=color)
        imwrite(os.path.join(save_path, 'individual_images',
                                '{}_flow_overlay_mask_color.png'.format(name_image)), flow_mask_overlay_color)
        
        flow_gt_mask_overlay_color = overlay_semantic_mask(rgb_flow_gt, 255 - mask_used.astype(np.uint8) * 255, color=color)
        imwrite(os.path.join(save_path, 'individual_images',
                                '{}_flow_gt_overlay_mask_color.png'.format(name_image)), flow_gt_mask_overlay_color)


def nearest_neighbor_flow(src_descriptor, trg_descriptor, ori_shape, mask1=None, mask2=None):
    B, C, H, W = src_descriptor.shape

    if mask1 is not None and mask2 is not None:
        resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=src_descriptor.shape[2:], mode='nearest')
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=trg_descriptor.shape[2:], mode='nearest')
        src_descriptor = src_descriptor * resized_mask1.repeat(1, src_descriptor.shape[1], 1, 1)
        trg_descriptor = trg_descriptor * resized_mask2.repeat(1, trg_descriptor.shape[1], 1, 1)
        # set where mask==0 a very large number
        src_descriptor[(src_descriptor.sum(1)==0).repeat(1, src_descriptor.shape[1], 1, 1)] = 100000
        trg_descriptor[(trg_descriptor.sum(1)==0).repeat(1, trg_descriptor.shape[1], 1, 1)] = 100000

    real_H, real_W = ori_shape
    long_edge = max(real_H, real_W)
    src_descriptor = src_descriptor.view(B, C, -1).permute(0, 2, 1).squeeze()
    trg_descriptor = trg_descriptor.view(B, C, -1).permute(0, 2, 1).squeeze()

    # Compute distance matrix using broadcasting and torch.cdist
    distances = torch.cdist(trg_descriptor, src_descriptor)

    # Find the indices of the minimum distances
    indices = torch.argmin(distances, dim=1).reshape(B, H, W)

    # Convert indices to coordinates
    trg_y = torch.div(indices, W).to(torch.float32)
    trg_x = torch.fmod(indices, W).to(torch.float32)

    # Create coordinate grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=src_descriptor.device), torch.arange(W, dtype=torch.float32, device=src_descriptor.device))

    # Compare target coordinates with source coordinate grid
    flow_x = trg_x - grid_x
    flow_y = trg_y - grid_y

    # Stack the flow fields together to form the final optical flow
    flow = torch.stack([flow_x, flow_y], dim=1)

    # Perform bilinear interpolation to adjust the optical flow from (60, 60) to (real_H, real_W)
    flow = F.interpolate(flow, size=(long_edge, long_edge), mode='bilinear', align_corners=False)
    flow *= torch.tensor([long_edge / 60.0, long_edge / 60.0], dtype=torch.float32, device=src_descriptor.device).view(1, 2, 1, 1)

    # Crop the flow field to the original image size
    if long_edge == real_H:
        flow = flow[:, :, :, (long_edge - real_W) // 2:(long_edge - real_W) // 2 + real_W]
    else:
        flow = flow[:, :, (long_edge - real_H) // 2:(long_edge - real_H) // 2 + real_H, :]

    return flow

def compute_flow(model, aug, source_img, target_img, save_path, batch_num=0, category=['car'], mask=False, dist='cos', real_size=960):
    if type(category) == str:
        category = [category]
    img_size = 840 if DINOV2 else 480
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
    stride = 14 if DINOV2 else 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    
    input_text = "a photo of "+category[-1][0] if TEXT_INPUT else None

    current_save_results = 0

    N = 1
    result = []

    for pair_idx in range(N):
        shape = source_img.shape[2:]
        # Load image 1
        img1=Image.fromarray(source_img.squeeze().numpy().transpose(1,2,0).astype(np.uint8))
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        # Load image 2
        img2=Image.fromarray(target_img.squeeze().numpy().transpose(1,2,0).astype(np.uint8))
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        with torch.no_grad():
            if not CO_PCA:
                if not ONLY_DINO:
                    img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                    img2_desc = process_features_and_mask(model, aug, img2_input, category[-1], input_text=input_text,  mask=mask).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

            else:
                if not ONLY_DINO:
                    features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text,  mask=False, raw=True)
                    features2 = process_features_and_mask(model, aug, img2_input, category[-1], input_text=input_text,  mask=mask, raw=True)
                    processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
                    img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                    img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)

                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet) # (1,1,3600,768)


            if dist == 'l1' or dist == 'l2':
                # normalize the features
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                if FUSE_DINO:
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            if FUSE_DINO and not ONLY_DINO:
                # cat two features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)
                
                img1_desc[...,:PCA_DIMS[0]]*=WEIGHT[0]
                img1_desc[...,PCA_DIMS[0]:PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[1]
                img1_desc[...,PCA_DIMS[1]+PCA_DIMS[0]:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[2]

                img2_desc[...,:PCA_DIMS[0]]*=WEIGHT[0]
                img2_desc[...,PCA_DIMS[0]:PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[1]
                img2_desc[...,PCA_DIMS[1]+PCA_DIMS[0]:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[2]

            if ONLY_DINO:
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino
            # logger.info(img1_desc.shape, img2_desc.shape)

            if DRAW_DENSE:
                mask1 = get_mask(model, aug, img1, category[0])
                mask2 = get_mask(model, aug, img2, category[-1])
                if ONLY_DINO or not FUSE_DINO:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                
                img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
                img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
                trg_dense_output, src_color_map = find_nearest_patchs(mask2, mask1, img2, img1, img2_desc_reshaped, img1_desc_reshaped, mask=mask)
                if current_save_results!=TOTAL_SAVE_RESULT:
                    if not os.path.exists(f'{save_path}/{category[0]}'):
                        os.makedirs(f'{save_path}/{category[0]}')
                    fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    ax1.axis('off')
                    ax2.axis('off')
                    ax1.imshow(src_color_map)
                    ax2.imshow(trg_dense_output)
                    fig_colormap.savefig(f'{save_path}/{category[0]}/{batch_num}_colormap.png')
                    plt.close(fig_colormap)
            
            if DRAW_SWAP:
                if not DRAW_DENSE:
                    mask1 = get_mask(model, aug, img1, category[0])
                    mask2 = get_mask(model, aug, img2, category[-1])

                if (ONLY_DINO or not FUSE_DINO) and not DRAW_DENSE:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                    
                img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
                img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
                trg_dense_output, src_color_map = find_nearest_patchs_replace(mask2, mask1, img2, img1, img2_desc_reshaped, img1_desc_reshaped, mask=mask, resolution=156)
                if current_save_results!=TOTAL_SAVE_RESULT:
                    if not os.path.exists(f'{save_path}/{category[0]}'):
                        os.makedirs(f'{save_path}/{category[0]}')
                    fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    ax1.axis('off')
                    ax2.axis('off')
                    ax1.imshow(src_color_map)
                    ax2.imshow(trg_dense_output)
                    fig_colormap.savefig(f'{save_path}/{category[0]}/{batch_num}_swap.png')
                    plt.close(fig_colormap)
            
            # compute the flow map based on the nearest neighbor
            # reshape the descriptors (1,dim,80,60)
            img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
            img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)

            # compute the flow map based on the nearest neighbor
            if MASK:
                mask1 = get_mask(model, aug, img1, category[0])
                mask2 = get_mask(model, aug, img2, category[-1])
                result = nearest_neighbor_flow(img1_desc_reshaped, img2_desc_reshaped, shape, mask1, mask2)
            else:
                result = nearest_neighbor_flow(img1_desc_reshaped, img2_desc_reshaped, shape)

    return result


def run_evaluation_semantic(model, aug, test_dataloader, device,
                            path_to_save=None, plot=False, plot_100=False, plot_ind_images=False):
    current_save_results = 0
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_0_05_list, pck_0_01_list, pck_0_1_list, pck_0_15_list = [], [], [], [], [], []
    smooth_est_list, smooth_gt_list = [], []
    eval_buf = {'cls_pck': dict(), 'vpvar': dict(), 'scvar': dict(), 'trncn': dict(), 'occln': dict()}

    # pck curve per image
    pck_thresholds = [0.01]
    pck_thresholds.extend(np.arange(0.05, 0.4, 0.05).tolist())
    pck_per_image_curve = np.zeros((len(pck_thresholds), len(test_dataloader)), np.float32)

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        category = mini_batch['category']

        if 'pckthres' in list(mini_batch.keys()):
            L_pck = mini_batch['pckthres'][0].float().item()
        else:
            raise ValueError('No pck threshold in mini_batch')

        flow_est = compute_flow(model, aug, source_img, target_img, batch_num=i_batch, save_path=path_to_save, category=category)

        if plot_ind_images or current_save_results < TOTAL_SAVE_RESULT:
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est,flow_gt , mask_used=mask_valid)
            current_save_results += 1

        smooth_est_list.append(get_smooth(flow_est,mask_valid).cpu().numpy())
        smooth_gt_list.append(get_smooth(flow_gt,mask_valid).cpu().numpy())

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_0_05_list.append(epe.le(0.05*L_pck).float().mean().item())
        pck_0_01_list.append(epe.le(0.01*L_pck).float().mean().item())
        pck_0_1_list.append(epe.le(0.1*L_pck).float().mean().item())
        pck_0_15_list.append(epe.le(0.15*L_pck).float().mean().item())
        for t in range(len(pck_thresholds)):
            pck_per_image_curve[t, i_batch] = epe.le(pck_thresholds[t]*L_pck).float().mean().item()

    epe_all = np.concatenate(epe_all_list)
    pck_0_05_dataset = np.mean(epe_all <= 0.05 * L_pck)
    pck_0_01_dataset = np.mean(epe_all <= 0.01 * L_pck)
    pck_0_1_dataset = np.mean(epe_all <= 0.1 * L_pck)
    pck_0_15_dataset = np.mean(epe_all <= 0.15 * L_pck)
    smooth_est_dataset = np.mean(smooth_est_list)
    smooth_gt_dataset = np.mean(smooth_gt_list)

    output = {'AEPE': np.mean(mean_epe_list), 'PCK_0_05_per_image': np.mean(pck_0_05_list),
              'PCK_0_01_per_image': np.mean(pck_0_01_list), 'PCK_0_1_per_image': np.mean(pck_0_1_list),
              'PCK_0_15_per_image': np.mean(pck_0_15_list),
              'PCK_0_01_per_dataset': pck_0_01_dataset, 'PCK_0_05_per_dataset': pck_0_05_dataset,
              'PCK_0_1_per_dataset': pck_0_1_dataset, 'PCK_0_15_per_dataset': pck_0_15_dataset,
              'pck_threshold_alpha': pck_thresholds, 'pck_curve_per_image': np.mean(pck_per_image_curve, axis=1).tolist()
              }
    logger.info("Validation EPE: %f, alpha=0_01: %f, alpha=0.05: %f" % (output['AEPE'], output['PCK_0_01_per_image'],
                                                                  output['PCK_0_05_per_image']))
    logger.info("smooth_est: %f, smooth_gt: %f" % (smooth_est_dataset, smooth_gt_dataset))

    for name in eval_buf.keys():
        output[name] = {}
        for cls in eval_buf[name]:
            if eval_buf[name] is not None:
                cls_avg = sum(eval_buf[name][cls]) / len(eval_buf[name][cls])
                output[name][cls] = cls_avg

    return output

def main(args):
    global MASK, SAMPLE, DIST, TOTAL_SAVE_RESULT, VER, CO_PCA, PCA_DIMS, SIZE, FUSE_DINO, DINOV2, MODEL_SIZE, DRAW_DENSE, TEXT_INPUT, DRAW_SWAP, ONLY_DINO, SEED, EDGE_PAD, WEIGHT
    MASK = args.MASK
    SAMPLE = args.SAMPLE
    DIST = args.DIST
    TOTAL_SAVE_RESULT = args.TOTAL_SAVE_RESULT
    VER = args.VER
    CO_PCA = args.CO_PCA
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
    TEXT_INPUT = args.TEXT_INPUT
    SEED = args.SEED
    WEIGHT = args.WEIGHT # corresponde to three groups for the sd features, and one group for the dino features

    if SAMPLE == 0:
        SAMPLE = None
    if DRAW_DENSE or DRAW_SWAP:
        TOTAL_SAVE_RESULT = SAMPLE
    if ONLY_DINO:
        FUSE_DINO = True
    if FUSE_DINO and not ONLY_DINO:
        DIST = "l2"
    else:
        DIST = "cos"

    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True

    model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=args.TIMESTEP, block_indices=tuple(INDICES))
    save_path=f'./results_tss/pck_tss_mask_{MASK}_dist_{DIST}_{args.TIMESTEP}{VER}_{MODEL_SIZE}_{SIZE}_copca_{CO_PCA}_{INDICES[0]}_{PCA_DIMS[0]}_{INDICES[1]}_{PCA_DIMS[1]}_{INDICES[2]}_{PCA_DIMS[2]}_text_{TEXT_INPUT}_sd_{not ONLY_DINO}_dino_{FUSE_DINO}'
    if EDGE_PAD:
        save_path += '_edge_pad'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logger = get_logger(save_path+'/result.log')

    logger.info(args)
    data_dir = "data/TSS_CVPR2016"

    start_time=time.time()

    class ArrayToTensor(object):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        def __init__(self, get_float=True):
            self.get_float = get_float

        def __call__(self, array):

            if not isinstance(array, np.ndarray):
                array = np.array(array)
            array = np.transpose(array, (2, 0, 1))
            # handle numpy array
            tensor = torch.from_numpy(array)
            # put it from HWC to CHW format
            if self.get_float:
                # carefull, this is not normalized to [0, 1]
                return tensor.float()
            else:
                return tensor

    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first
    output = {}
    for sub_data in ['FG3DCar', 'JODS', 'PASCAL']:
        test_set = TSSDataset(os.path.join(data_dir, sub_data),
                                                    source_image_transform=input_transform,
                                                    target_image_transform=input_transform, flow_transform=target_transform,
                                                    co_transform=co_transform,
                                                    num_samples=SAMPLE)
        test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
        results = run_evaluation_semantic(model,aug, test_dataloader, device='cuda', path_to_save=save_path+'/'+sub_data, plot_ind_images=DRAW_SWAP)
        output[sub_data] = results

    end_time=time.time()
    minutes, seconds = divmod(end_time-start_time, 60)
    logger.info(f"Time: {minutes:.0f}m {seconds:.0f}s")
    torch.save(output, save_path+'/result.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--MASK', action='store_true', default=False)
    parser.add_argument('--SAMPLE', type=int, default=0)
    parser.add_argument('--DIST', type=str, default='l2')
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=5)
    parser.add_argument('--VER', type=str, default="v1-5")
    parser.add_argument('--CO_PCA', type=bool, default=True)
    parser.add_argument('--PCA_DIMS', nargs=3, type=int, default=[256, 256, 256])
    parser.add_argument('--TIMESTEP', type=int, default=100)
    parser.add_argument('--SIZE', type=int, default=960)
    parser.add_argument('--INDICES', nargs=4, type=int, default=[2,5,8,11])
    parser.add_argument('--WEIGHT', nargs=4, type=float, default=[1,1,1,1])
    parser.add_argument('--EDGE_PAD', action='store_true', default=False)

    parser.add_argument('--NOT_FUSE', action='store_true', default=False)
    parser.add_argument('--ONLY_DINO', action='store_true', default=False)
    parser.add_argument('--DINOV1', action='store_true', default=False)
    parser.add_argument('--MODEL_SIZE', type=str, default='base')

    parser.add_argument('--DRAW_DENSE', action='store_true', default=False)
    parser.add_argument('--DRAW_SWAP', action='store_true', default=False)
    parser.add_argument('--TEXT_INPUT', action='store_true', default=False)

    args = parser.parse_args()
    main(args)