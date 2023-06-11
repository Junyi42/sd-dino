import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import faiss
import cv2
import os
from matplotlib.patches import ConnectionPatch

def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def find_nearest_patchs(mask1, mask2, image1, image2, features1, features2, mask=False, resolution=None, edit_image=None):
    def polar_color_map(image_shape):
        h, w = image_shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Find the center of the mask
        mask=mask2.cpu()
        mask_center = np.array(np.where(mask > 0))
        mask_center = np.round(np.mean(mask_center, axis=1)).astype(int)
        mask_center_y, mask_center_x = mask_center

        # Calculate distance and angle based on mask_center
        xx_shifted, yy_shifted = xx - x[mask_center_x], yy - y[mask_center_y]
        max_radius = np.sqrt(h**2 + w**2) / 2
        radius = np.sqrt(xx_shifted**2 + yy_shifted**2) * max_radius
        angle = np.arctan2(yy_shifted, xx_shifted) / (2 * np.pi) + 0.5

        angle = 0.2 + angle * 0.6  # Map angle to the range [0.25, 0.75]
        radius = np.where(radius <= max_radius, radius, max_radius)  # Limit radius values to the unit circle
        radius = 0.2 + radius * 0.6 / max_radius  # Map radius to the range [0.1, 1]

        return angle, radius
    
    if resolution is not None: # resize the feature map to the resolution
        features1 = F.interpolate(features1, size=resolution, mode='bilinear')
        features2 = F.interpolate(features2, size=resolution, mode='bilinear')
    
    # resize the image to the shape of the feature map
    resized_image1 = resize(image1, features1.shape[2], resize=True, to_pil=False)
    resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)

    if mask: # mask the features
        resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=features1.shape[2:], mode='nearest')
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[2:], mode='nearest')
        features1 = features1 * resized_mask1.repeat(1, features1.shape[1], 1, 1)
        features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
        # set where mask==0 a very large number
        features1[(features1.sum(1)==0).repeat(1, features1.shape[1], 1, 1)] = 100000
        features2[(features2.sum(1)==0).repeat(1, features2.shape[1], 1, 1)] = 100000

    features1_2d = features1.reshape(features1.shape[1], -1).permute(1, 0).cpu().detach().numpy()
    features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0).cpu().detach().numpy()

    features1_2d = torch.tensor(features1_2d).to("cuda")
    features2_2d = torch.tensor(features2_2d).to("cuda")
    resized_image1 = torch.tensor(resized_image1).to("cuda").float()
    resized_image2 = torch.tensor(resized_image2).to("cuda").float()

    mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image1.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)

    # Mask the images
    resized_image1 = resized_image1 * mask1.unsqueeze(-1).repeat(1, 1, 3)
    resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
    # Normalize the images to the range [0, 1]
    resized_image1 = (resized_image1 - resized_image1.min()) / (resized_image1.max() - resized_image1.min())
    resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())

    angle, radius = polar_color_map(resized_image2.shape)

    angle_mask = angle * mask2.cpu().numpy()
    radius_mask = radius * mask2.cpu().numpy()

    hsv_mask = np.zeros(resized_image2.shape, dtype=np.float32)
    hsv_mask[:, :, 0] = angle_mask
    hsv_mask[:, :, 1] = radius_mask
    hsv_mask[:, :, 2] = 1

    rainbow_mask2 = cv2.cvtColor((hsv_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR) / 255

    if edit_image is not None:
        rainbow_mask2 = cv2.imread(edit_image, cv2.IMREAD_COLOR)
        rainbow_mask2 = cv2.cvtColor(rainbow_mask2, cv2.COLOR_BGR2RGB) / 255
        rainbow_mask2 = cv2.resize(rainbow_mask2, (resized_image2.shape[1], resized_image2.shape[0]))

    # Apply the rainbow mask to image2
    rainbow_image2 = rainbow_mask2 * mask2.cpu().numpy()[:, :, None]

    # Create a white background image
    background_color = np.array([1, 1, 1], dtype=np.float32)
    background_image = np.ones(resized_image2.shape, dtype=np.float32) * background_color

    # Apply the rainbow mask to image2 only in the regions where mask2 is 1
    rainbow_image2 = np.where(mask2.cpu().numpy()[:, :, None] == 1, rainbow_mask2, background_image)
    
    nearest_patches = []

    distances = torch.cdist(features1_2d, features2_2d)
    nearest_patch_indices = torch.argmin(distances, dim=1)
    nearest_patches = torch.index_select(torch.tensor(rainbow_mask2).cuda().reshape(-1, 3), 0, nearest_patch_indices)

    nearest_patches_image = nearest_patches.reshape(resized_image1.shape)
    rainbow_image2 = torch.tensor(rainbow_image2).to("cuda")

    # TODO: upsample the nearest_patches_image to the resolution of the original image
    # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
    # rainbow_image2 = F.interpolate(rainbow_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

    nearest_patches_image = (nearest_patches_image).cpu().numpy()
    resized_image2 = (rainbow_image2).cpu().numpy()

    return nearest_patches_image, resized_image2


def find_nearest_patchs_replace(mask1, mask2, image1, image2, features1, features2, mask=False, resolution=128, draw_gif=False, save_path=None, gif_reverse=False):
    
    if resolution is not None: # resize the feature map to the resolution
        features1 = F.interpolate(features1, size=resolution, mode='bilinear')
        features2 = F.interpolate(features2, size=resolution, mode='bilinear')
    
    # resize the image to the shape of the feature map
    resized_image1 = resize(image1, features1.shape[2], resize=True, to_pil=False)
    resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)

    if mask: # mask the features
        resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=features1.shape[2:], mode='nearest')
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[2:], mode='nearest')
        features1 = features1 * resized_mask1.repeat(1, features1.shape[1], 1, 1)
        features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
        # set where mask==0 a very large number
        features1[(features1.sum(1)==0).repeat(1, features1.shape[1], 1, 1)] = 100000
        features2[(features2.sum(1)==0).repeat(1, features2.shape[1], 1, 1)] = 100000
    
    features1_2d = features1.reshape(features1.shape[1], -1).permute(1, 0)
    features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0)

    resized_image1 = torch.tensor(resized_image1).to("cuda").float()
    resized_image2 = torch.tensor(resized_image2).to("cuda").float()

    mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image1.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)

    # Mask the images
    resized_image1 = resized_image1 * mask1.unsqueeze(-1).repeat(1, 1, 3)
    resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
    # Normalize the images to the range [0, 1]
    resized_image1 = (resized_image1 - resized_image1.min()) / (resized_image1.max() - resized_image1.min())
    resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())

    distances = torch.cdist(features1_2d, features2_2d)
    nearest_patch_indices = torch.argmin(distances, dim=1)
    nearest_patches = torch.index_select(resized_image2.cuda().clone().detach().reshape(-1, 3), 0, nearest_patch_indices)

    nearest_patches_image = nearest_patches.reshape(resized_image1.shape)

    if draw_gif:
        assert save_path is not None, "save_path must be provided when draw_gif is True"
        img_1 = resize(image1, features1.shape[2], resize=True, to_pil=True)
        img_2 = resize(image2, features2.shape[2], resize=True, to_pil=True)
        mapping = torch.zeros((img_1.size[1], img_1.size[0], 2))
        for i in range(len(nearest_patch_indices)):
            mapping[i // img_1.size[0], i % img_1.size[0]] = torch.tensor([nearest_patch_indices[i] // img_2.size[0], nearest_patch_indices[i] % img_2.size[0]])
        animate_image_transfer(img_1, img_2, mapping, save_path) if gif_reverse else animate_image_transfer_reverse(img_1, img_2, mapping, save_path)

    # TODO: upsample the nearest_patches_image to the resolution of the original image
    # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
    # resized_image2 = F.interpolate(resized_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

    nearest_patches_image = (nearest_patches_image).cpu().numpy()
    resized_image2 = (resized_image2).cpu().numpy()

    return nearest_patches_image, resized_image2

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

def pairwise_sim(x: torch.Tensor, y: torch.Tensor, p=2, normalize=False) -> torch.Tensor:
    # compute similarity based on euclidean distances
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
    result_list=[]
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)
        result_list.append(torch.nn.PairwiseDistance(p=p)(token, y)*(-1))
    return torch.stack(result_list, dim=2)

def draw_correspondences_gathered(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                        image1: Image.Image, image2: Image.Image) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: a figure of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    # plot a subfigure put image1 in the top, image2 in the bottom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)

    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

    return fig

def draw_correspondences_lines(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], 
                               gt_points2: List[Tuple[float, float]], image1: Image.Image, 
                               image2: Image.Image, threshold=None) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param gt_points2: a list of ground truth (y, x) coordinates of image2.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :param threshold: distance threshold to determine correct matches.
    :return: a figure of images with marked points and lines between them showing correspondence.
    """

    points2=points2.cpu().numpy()
    gt_points2=gt_points2.cpu().numpy()

    def compute_correct():
        alpha = torch.tensor([0.1, 0.05, 0.01])
        correct = torch.zeros(len(alpha))
        err = (torch.tensor(points2) - torch.tensor(gt_points2)).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        correct = err < threshold.unsqueeze(-1) if len(threshold.shape)==1 else err < threshold
        return correct

    correct = compute_correct()[0]
    # print(correct.shape, len(points1)) 

    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.set_xlim(0, image1.size[0])
    ax1.set_ylim(image1.size[1], 0)
    ax2.set_xlim(0, image2.size[0])
    ax2.set_ylim(image2.size[1], 0)

    for i, (point1, point2) in enumerate(zip(points1, points2)):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=colors[i], edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=colors[i], edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=colors[i], edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=colors[i], edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

        # Draw lines
        color = 'blue' if correct[i].item() else 'red'
        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=color, linewidth=1.5)
        ax2.add_artist(con)

    return fig

def co_pca(features1, features2, dim=[128,128,128]):
    
    processed_features1 = {}
    processed_features2 = {}
    s5_size = features1['s5'].shape[-1]
    s4_size = features1['s4'].shape[-1]
    s3_size = features1['s3'].shape[-1]
    # Get the feature tensors
    s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
    s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
    s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)

    s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
    s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
    s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)
    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [[s5_1, s5_2], [s4_1, s4_2], [s3_1, s3_2]]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        

        # Split the features
        processed_features1[name] = features[:, :, :features.shape[-1] // 2] # Bx(d)x(t_x)
        processed_features2[name] = features[:, :, features.shape[-1] // 2:] # Bx(d)x(t_y)

    # reshape the features
    processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size, s5_size)
    processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size, s4_size)
    processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size, s3_size)

    processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size, s5_size)
    processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size, s4_size)
    processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size, s3_size)

    # Upsample s5 spatially by a factor of 2
    processed_features1['s5'] = F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features2['s5'] = F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1['s5'] = torch.cat([processed_features1['s4'], processed_features1['s5']], dim=1)
    processed_features2['s5'] = torch.cat([processed_features2['s4'], processed_features2['s5']], dim=1)

    # Set s3 as the new s4
    processed_features1['s4'] = processed_features1['s3']
    processed_features2['s4'] = processed_features2['s3']

    # Remove s3 from the features dictionary
    processed_features1.pop('s3')
    processed_features2.pop('s3')

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat([processed_features1['s4'], F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features2_gether_s4_s5 = torch.cat([processed_features2['s4'], F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features1_gether_s4_s5, features2_gether_s4_s5

def animate_image_transfer(image1, image2, mapping, output_path):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # # Load your two images
    # image1 = Image.open(image1_path)
    # image2 = Image.open(image2_path)

    # Ensure the two images are the same size
    assert image1.size == image2.size, "Images must be the same size."
    rec_size = 2
    # Convert the images into numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Retrieve the width and height of the images
    height, width, _ = image1_array.shape

    # Assume we have a mapping list
    mapping = mapping.cpu().numpy()

    # We add a column of white pixels between the two images
    gap = width // 10

    # Create a canvas with a width that is the sum of the widths of the two images and the gap. 
    # The height is the same as the height of the images.
    fig, ax = plt.subplots(figsize=((2 * width + gap) / 200, height / 200), dpi=300)

    # Remove the axes
    ax.axis('off')

    # Create an image object, initializing it as entirely white
    combined_image = np.ones((height, 2 * width + gap, 3), dtype=np.uint8) * 255

    # Place image1 on the left, image2 on the right, with a gap in the middle
    combined_image[:, :width] = image1_array
    combined_image[:, width + gap:] = image2_array

    img_obj = ax.imshow(combined_image)

    # For each frame of the computation and animation, we need to know the start and target positions of each pixel
    starts = np.mgrid[:height, :width].reshape(2, -1).T
    targets = np.array([mapping[i, j] for i in range(height) for j in range(width)]) + [0, width + gap]

    # To better display the animation, we divide the pixel movement into several frames
    num_frames = 30

    def calculate_path(start, target, num_frames):
        """Calculate the path of a pixel from start to target over num_frames."""
        # Generate linear values from 0 to 1
        t = np.linspace(0, 1, num_frames)

        # Apply the quadratic easing out function (starts fast, then slows down)
        t = 1 - (1 - t) ** 2

        # Calculate the path
        path = start + t[:, np.newaxis] * (target - start)

        return path

    def update(frame):
        # At the start of each frame, we initialize the canvas with image1 on the left, image2 on the right, and white in the middle
        combined_image.fill(255)
        combined_image[:, :width] = image1_array
        combined_image[:, width + gap:] = image2_array
        # In each frame, we move a small portion of pixels from the left image to the right image
        # This gives a better view of how the pixels move
        if frame >= num_frames - 1:
            frame = num_frames - 1
        for i in range(height):
            for j in range(width):
                # Calculate the current pixel's position
                start = starts[i * width + j]
                target = targets[i * width + j]
                # If the mapped target position is greater than 0, move the pixel, otherwise keep it the same
                if target[0] > 0 and target[1] > 0:
                    position = calculate_path(start, target, num_frames)[frame]
                    # Copy the current pixel's color to the new position
                    combined_image[int(position[0])-rec_size//2:int(position[0])-rec_size//2+rec_size, int(position[1])-rec_size//2:int(position[1])-rec_size//2+rec_size] = image1_array[i, j]
        img_obj.set_array(combined_image)  # Update the displayed image
        return img_obj,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames + 30, blit=True)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Save the animation
    ani.save(output_path, writer='pillow', fps=30)
    # save mapping
    np.save(output_path[:-4]+'.npy', mapping)


def animate_image_transfer_reverse(image1, image2, mapping, output_path):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # # Load your two images
    # image1 = Image.open(image1_path)
    # image2 = Image.open(image2_path)

    # Ensure the two images are the same size
    assert image1.size == image2.size, "Images must be the same size."
    # rec_size = 2
    # Convert the images into numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Retrieve the width and height of the images
    height, width, _ = image1_array.shape

    # Assume we have a mapping list
    mapping = mapping.cpu().numpy()

    # We add a column of white pixels between the two images
    gap = width // 10

    # Create a canvas with a width that is the sum of the widths of the two images and the gap. 
    # The height is the same as the height of the images.
    fig, ax = plt.subplots(figsize=((2 * width + gap) / 200, height / 200), dpi=300)

    # Remove the axes
    ax.axis('off')

    # Create an image object, initializing it as entirely white
    combined_image = np.ones((height, 2 * width + gap, 3), dtype=np.uint8) * 255

    # Place image1 on the left, image2 on the right, with a gap in the middle
    combined_image[:, :width] = image2_array
    combined_image[:, width + gap:] = image1_array

    img_obj = ax.imshow(combined_image)

    # For each frame of the computation and animation, we need to know the start and target positions of each pixel
    starts = np.mgrid[:height, :width].reshape(2, -1).T + [0, width + gap]
    targets = np.array([mapping[i, j] for i in range(height) for j in range(width)])

    # To better display the animation, we divide the pixel movement into several frames
    num_frames = 30

    def calculate_path(start, target, num_frames):
        """Calculate the path of a pixel from start to target over num_frames."""
        # Generate linear values from 0 to 1
        t = np.linspace(1, 0, num_frames)

        # Apply the quadratic easing out function (starts fast, then slows down)
        t = 1 - (1 - t) ** 2

        # Calculate the path
        path = start + t[:, np.newaxis] * (target - start)

        return path

    def update(frame):
        # At the start of each frame, we initialize the canvas with image1 on the left, image2 on the right, and white in the middle
        combined_image.fill(255)
        combined_image[:, :width] = image2_array
        combined_image[:, width + gap:] = image1_array
        # In each frame, we move a small portion of pixels from the left image to the right image
        # This gives a better view of how the pixels move
        if frame >= num_frames - 1:
            frame = num_frames - 1
        if frame >= num_frames // 6 * 5:
            rec_size = 1
        else:
            rec_size = 2
        for i in range(height):
            for j in range(width):
                # Calculate the current pixel's position
                start = starts[i * width + j]
                target = targets[i * width + j]
                # If the mapped target position is greater than 0, move the pixel, otherwise keep it the same
                if target[0] > 0 and target[1] > 0:
                    position = calculate_path(start, target, num_frames)[frame]
                    # Copy the current pixel's color to the new position
                    combined_image[int(position[0])-rec_size//2:int(position[0])-rec_size//2+rec_size, int(position[1])-rec_size//2:int(position[1])-rec_size//2+rec_size] = image2_array[int(mapping[i, j][0]), int(mapping[i, j][1])]
        img_obj.set_array(combined_image)  # Update the displayed image
        return img_obj,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames + 30, blit=True)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Save the animation
    ani.save(output_path, writer='pillow', fps=30)
    # save the maping
    np.save(output_path[:-4]+'.npy', mapping)