import time
from matplotlib import projections
import torch
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
import src.losses as losses
from src.losses import ChamferDistanceLoss
import numpy as np
import matplotlib.pyplot as plt
import hydra
import os
from omegaconf import DictConfig
from pytorch3d.structures import Pointclouds, Volumes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    VolumeRenderer,
    FoVPerspectiveCameras,
    ImplicitRenderer
)
import yaml
from pytorch3d.ops import sample_points_from_meshes
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.datasets import collate_batched_meshes


cd_loss = ChamferDistanceLoss()

def calculate_loss(predictions, ground_truth, cfg):
    if cfg.dtype == 'voxel':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif cfg.dtype == 'point':
        loss = cd_loss(predictions, ground_truth)
    elif cfg.dtype == 'mesh':
        sample_trg = sample_points_from_meshes(ground_truth, cfg.n_points)
        sample_pred = sample_points_from_meshes(predictions, cfg.n_points)

        loss_reg = cd_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)
        loss_edge = losses.edge_loss(predictions)
        loss_normal = losses.normal_loss(predictions)

        loss = cfg.w_chamfer * loss_reg + cfg.w_smooth * loss_smooth \
            + cfg.w_edge * loss_edge + cfg.w_normal * loss_normal
    return loss


def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points[0].clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


def visualization(prediction_3d, cfg):
    if cfg.dtype == 'point':
        R, T = look_at_view_transform(45, 45, 45)
        cameras = FoVOrthographicCameras(device='cuda', R=R, T=T, znear=0.01)
        raster_settings = PointsRasterizationSettings(
            image_size=256, 
            radius = 0.003,
            points_per_pixel = 10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        features = torch.ones_like(prediction_3d[0])
        point_cloud = Pointclouds(points=[prediction_3d[0]], features=[features])
        rend = renderer(point_cloud)
        prediction_3d = prediction_3d.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(prediction_3d[0,...,0], prediction_3d[0,...,1], prediction_3d[0,...,2], c='r', marker='.')
        plt.show()

    elif cfg.dtype == 'voxel':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(prediction_3d.detach().cpu().numpy()[0] > 0.4, edgecolor='k')
        plt.show()
        R, T = look_at_view_transform(3, 0, 90)
        cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T)
        raysampler = NDCMultinomialRaysampler(
            image_width=256,
            image_height=256,
            n_pts_per_ray=150,
            min_depth=0.1,
            max_depth=3.0,
        )
        raymarcher = EmissionAbsorptionRaymarcher()
        renderer = VolumeRenderer(
            raysampler=raysampler, raymarcher=raymarcher,
        )
        prediction_3d = prediction_3d[0].view(-1, 1, 33, 33, 33)
        prediction_3d[prediction_3d<0.4] = 0.0
        prediction_3d[prediction_3d>=0.4] = 1.0
        features = torch.ones(1, 3, 33, 33, 33).cuda()
        volumes = Volumes(
            densities=prediction_3d,
            features=features,
            voxel_size=3.0/33
        )
        rend, _ = renderer(cameras=cameras, volumes=volumes)[0].split([3, 1], dim=-1)
    
    elif cfg.dtype == 'mesh':
        plot_pointcloud(prediction_3d)
        rend = None

    return rend

@hydra.main(config_path="configs/", config_name="config.yml")
def evaluate_model(cfg: DictConfig):
    shapenetdb = ShapeNetDB(cfg.data_dir, cfg.dtype)

    if cfg.dtype=='mesh':
        loader = torch.utils.data.DataLoader(
            shapenetdb,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_batched_meshes)
    else:
        loader = torch.utils.data.DataLoader(
            shapenetdb,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_loss = []

    if cfg.load_eval_checkpoint:
        checkpoint = torch.load(f'{cfg.base_dir}/checkpoint_{cfg.dtype}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        if cfg.dtype == 'mesh':
            ground_truth_3d = next(eval_loader)
            images_gt = torch.Tensor([t.numpy() for t in ground_truth_3d['image']]).cuda()
            ground_truth_3d = ground_truth_3d['mesh'].cuda()
            
        else:
            images_gt, ground_truth_3d, _ = next(eval_loader)
            images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()

        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, cfg)
        # torch.save(prediction_3d.detach().cpu(), f'{cfg.base_dir}/pre_point_cloud.pt')

        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg).cpu().item()

        # TODO:
        if (step % cfg.vis_freq) == 0:
            # visualization block
            rend = visualization(prediction_3d, cfg)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(rend[0].detach().cpu().numpy())
            # plt.axis("off")
            # plt.savefig(f'{step}_{cfg.dtype}.png')
            plt.imsave(f'{step}_image_gt.png', images_gt[0].detach().cpu().numpy())

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        avg_loss.append(loss)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); eva_loss: %.3f" % (step, cfg.max_iter, total_time, read_time, iter_time, torch.tensor(avg_loss).mean()))

    print('Done!')

if __name__ == '__main__':
    evaluate_model()