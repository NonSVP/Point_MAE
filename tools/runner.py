import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
import cv2
import numpy as np

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    # DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

# visualization
def test(base_model, test_dataloader, args, config, logger = None):
    base_model.eval()  # set model to eval mode
    
    # Ensure the output directory exists
    if not os.path.exists('./vis'):
        os.makedirs('./vis')

    with torch.no_grad():
        for idx, batch_data in enumerate(test_dataloader):
            # FIX 1: Flexible Unpacking
            # Handles (taxonomy_ids, model_ids, data) or (model_ids, data)
            if len(batch_data) == 3:
                taxonomy_ids, model_ids, data = batch_data
            elif len(batch_data) == 2:
                model_ids, data = batch_data
                taxonomy_ids = ["custom"] * len(model_ids)
            else:
                data = batch_data
                model_ids = [str(idx)] * data.shape[0]
                taxonomy_ids = ["custom"] * data.shape[0]

            # FIX 2: Support MyCustomDataset
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet' or dataset_name == 'MyCustomDataset':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, config.dataset.test.others.npoints)
            else:
                # Fallback for other custom naming
                points = data.cuda() if torch.is_tensor(data) else data[0].cuda()

            # Set a standard viewing angle for custom data (a: elevation, b: azimuth)
            a, b = 30, 45 

            # Forward pass: dense_points is the full recon, vis_points is the masked input
            dense_points, vis_points, centers, gt_serialized = base_model(points, vis=True)
            
            final_image = []
            data_path = os.path.join('./vis', f'{taxonomy_ids[0]}_{idx}')
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            # --- 1. Ground Truth (Full Object) ---
            points_np = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'gt.txt'), points_np, delimiter=';')
            gt_img = misc.get_ptcloud_img(points_np, a, b)
            # Crop to focus on the object
            final_image.append(gt_img[150:650, 150:675, :])

            # --- 2. Visible Points (What the model saw - 40%) ---
            vis_points_np = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points_np, delimiter=';')
            vis_img = misc.get_ptcloud_img(vis_points_np, a, b)
            final_image.append(vis_img[150:650, 150:675, :])

            # --- 3. Reconstructed Points (What the AI filled in) ---
            dense_points_np = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points_np, delimiter=';')
            recon_img = misc.get_ptcloud_img(dense_points_np, a, b)
            final_image.append(recon_img[150:650, 150:675, :])

            gt_serialized = gt_serialized.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'gt_serialized.txt'), gt_serialized, delimiter=';')

            # Concatenate images side-by-side: [GT | Masked | Reconstruction]
            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, 'plot.jpg')
            cv2.imwrite(img_path, img)
            import matplotlib.pyplot as plt
            plt.close('all') # This frees the bitmap memory

            # Limit total visualizations to prevent disk overflow
            if idx > 1500:
                break

        print_log(f"Visualization completed. Results saved in ./vis/", logger=logger)
        return