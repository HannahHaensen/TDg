# TDg.py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Pure thermal single-modality + Depth estimation joint optimization (TDg)

import os
import torch
import time
import sys
import uuid
import cv2
import numpy as np
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from utils.loss_utils import l1_loss, ssim, smoothness_loss
from gaussian_renderer import render, render_depth, network_gui
from scene import Scene_2, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _try_load_image_from_disk(path: str):
    if not os.path.isfile(path):
        return None
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if data is None:
        return None
    if data.ndim == 2:
        img = torch.from_numpy(data.astype(np.float32))
        if img.max() > 1.0:
            # Assume 16-bit image format
            if np.iinfo(data.dtype).max > 255:
                 img = img / 65535.0
            else:
                 img = img / 255.0
        return img.unsqueeze(0)
    else:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32)
        if data.max() > 1.0:
            data = data / 255.0
        t = torch.from_numpy(np.transpose(data, (2, 0, 1)))
        return t


def _maybe_resize_like(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if img.shape[-2:] == ref.shape[-2:]:
        return img
    img = img.unsqueeze(0)
    out = torch.nn.functional.interpolate(img, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return out.squeeze(0)


def _fetch_depth_image_from_depthesti(dataset, cam, pred_Tstar: torch.Tensor):
    device = pred_Tstar.device
    image_name = getattr(cam, "image_name", None)
    if image_name is None and hasattr(cam, "image_path"):
        image_name = os.path.basename(cam.image_path)

    depth_img = None
    base = dataset.source_path if hasattr(dataset, "source_path") else None
    if base and image_name:
        depthesti_paths = [
            os.path.join(base, "depthesti", "train", image_name),
            os.path.join(base, "depthesti", "test", image_name),
            os.path.join(base, "depthesti", image_name),
        ]

        for p in depthesti_paths:
            if not p.lower().endswith('.png'):
                p = os.path.splitext(p)[0] + '.png'

            t = _try_load_image_from_disk(p)
            if t is not None:
                depth_img = t
                break

    if depth_img is not None:
        depth_img = _maybe_resize_like(depth_img, pred_Tstar).to(device)

    return depth_img


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Initialize model and scene
    gaussians_2 = GaussianModel(dataset.sh_degree)
    scene_2 = Scene_2(dataset, gaussians_2, load_iteration=0, shuffle=True)
    gaussians_2.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_2.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack_2 = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Define progressive optimization milestones (Section 3.1.4 of the paper)
    t_start = 1
    t_end = int(opt.iterations * 0.5)  # Phase out depth constraint at 50% of total iterations

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes_1 = None
                net_image_bytes_2 = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image_1 = render(custom_cam, gaussians_2, pipe, background, scaling_modifer)["render"]
                    net_image_bytes_1 = memoryview(
                        (torch.clamp(net_image_1, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    net_image_2 = render(custom_cam, gaussians_2, pipe, background, scaling_modifer)["render"]
                    net_image_bytes_2 = memoryview(
                        (torch.clamp(net_image_2, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes_1, dataset.source_path)
                network_gui.send(net_image_bytes_2, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians_2.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians_2.oneupSHdegree()

        if not viewpoint_stack_2:
            viewpoint_stack_2 = scene_2.getTrainCameras().copy()

        viewpoint_cam_2 = viewpoint_stack_2.pop(randint(0, len(viewpoint_stack_2) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg_2 = render(viewpoint_cam_2, gaussians_2, pipe, bg)
        image_2 = render_pkg_2["render"]
        viewspace_point_tensor_2 = render_pkg_2["viewspace_points"]
        visibility_filter_2 = render_pkg_2["visibility_filter"]
        radii_2 = render_pkg_2["radii"]

        smoothloss_thermal = smoothness_loss(image_2)
        gt_image_2 = viewpoint_cam_2.original_image.cuda()
        Ll1_2 = l1_loss(image_2, gt_image_2)

        # ----------------------------------------------------
        # Depth Rendering Loss Calculation
        # ----------------------------------------------------
        loss_depth_render = torch.tensor(0.0, device="cuda")
        depth_img = None
        
        # Dynamically compute depth decay weight w_decay(t)
        if iteration <= t_start:
            w_decay = 1.0
        elif iteration >= t_end:
            w_decay = 0.0
        else:
            w_decay = 1.0 - (iteration - t_start) / (t_end - t_start)

        # Compute depth loss only when weight > 0 to save computation
        if hasattr(opt, 'lambda_depth_render') and opt.lambda_depth_render > 0 and w_decay > 0:
            depth_img = _fetch_depth_image_from_depthesti(dataset, viewpoint_cam_2, image_2)

            if depth_img is not None:
                try:
                    # Render depth map from 3D Gaussians
                    with torch.no_grad():
                        depth_render_pkg = render_depth(viewpoint_cam_2, gaussians_2, pipe, bg)
                        rendered_depth_norm = depth_render_pkg["render_vis"].detach()

                    # Min-max normalization on GT depth map (Eq. 4 in the paper)
                    valid_mask_gt = depth_img > 0
                    if valid_mask_gt.any():
                        d_min = depth_img[valid_mask_gt].min()
                        d_max = depth_img[valid_mask_gt].max()
                        if (d_max - d_min) > 1e-8:
                            depth_img_norm = (depth_img - d_min) / (d_max - d_min)
                            depth_img_norm[~valid_mask_gt] = 0  
                        else:
                            depth_img_norm = torch.zeros_like(depth_img)
                    else:
                        depth_img_norm = torch.zeros_like(depth_img)

                    # Match spatial dimensions
                    depth_img_resized_norm = _maybe_resize_like(depth_img_norm, rendered_depth_norm)

                    # Calculate L1 + SSIM loss
                    loss_depth_render = l1_loss(rendered_depth_norm, depth_img_resized_norm) + \
                                        (1.0 - ssim(rendered_depth_norm, depth_img_resized_norm))

                except Exception as e:
                    print(f"Unexpected error in depth rendering loss calculation: {e}")
                    loss_depth_render = torch.tensor(0.0, device="cuda")

        # ----------------------------------------------------
        # Total Loss Calculation
        # ----------------------------------------------------
        loss_2 = (1.0 - opt.lambda_dssim) * Ll1_2 + opt.lambda_dssim * (
                    1.0 - ssim(image_2, gt_image_2)) + 0.6 * smoothloss_thermal

        # Apply progressively decaying depth loss weight
        if hasattr(opt, 'lambda_depth_render') and opt.lambda_depth_render > 0:
            loss_2 = loss_2 + w_decay * opt.lambda_depth_render * loss_depth_render

        total_loss = loss_2
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.7f}"}
                if hasattr(opt, 'lambda_depth_render') and opt.lambda_depth_render > 0 and depth_img is not None:
                    postfix["DepthRender"] = f"{loss_depth_render.item():.4f}"
                    postfix["w_decay"] = f"{w_decay:.2f}" # Monitor decay coefficient
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1_2, total_loss, Ll1_2, total_loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene_2, render, (pipe, background))

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene_2.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians_2.max_radii2D[visibility_filter_2] = torch.max(
                    gaussians_2.max_radii2D[visibility_filter_2], radii_2[visibility_filter_2])
                gaussians_2.add_densification_stats(viewspace_point_tensor_2, visibility_filter_2)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_2.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_2.cameras_extent,
                                                  size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians_2.reset_opacity()

            if iteration < opt.iterations:
                gaussians_2.optimizer.step()
                gaussians_2.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians_2.capture(), iteration), scene_2.model_path + f"/chkpnt_2{iteration}.pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
        return None


def training_report(tb_writer, iteration, Ll1_1, loss_1, Ll1_2, loss_2, l1_loss_func, elapsed, testing_iterations,
                    scene_2, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss/l1_loss', Ll1_2.item(), iteration)
        tb_writer.add_scalar('train_loss/total_loss', loss_2.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # Set --lambda-depth-render > 0 to enable TDg method (e.g., 1.0)
    parser.add_argument("--lambda-depth-render", type=float, default=0.0,
                        help="Weight for depth rendering loss (L1 + SSIM with depthesti depth). Set > 0 to enable TDg.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")