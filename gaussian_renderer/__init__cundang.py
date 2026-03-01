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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


# def render_depth(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
#     """
#     Render depth map from the scene.
#
#     Background tensor (bg_color) must be on GPU!
#     Returns single-channel depth image.
#     """
#     # 确保所有张量都在正确的设备上
#     device = bg_color.device
#
#     # Set up rasterization configuration (same as color rendering)
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug
#     )
#
#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#
#     means3D = pc.get_xyz
#     opacity = pc.get_opacity
#
#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation
#
#     # Calculate depth value for each Gaussian point in camera space
#     # Transform points to camera space
#     points_homogeneous = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
#     points_camera = torch.matmul(points_homogeneous, viewpoint_camera.world_view_transform.T)
#
#     # Calculate depth value (z / w)
#     depths = points_camera[:, 2] / points_camera[:, 3]
#
#     # 确保深度值在合理范围内
#     depths = torch.clamp(depths, min=-1000.0, max=1000.0)  # 添加合理的范围限制
#
#     # 使用百分位数而不是min/max来避免异常值
#     depth_min = torch.quantile(depths, 0.01)
#     depth_max = torch.quantile(depths, 0.99)
#
#     if depth_max - depth_min > 1e-6:
#         depths_normalized = (depths - depth_min) / (depth_max - depth_min)
#     else:
#         depths_normalized = torch.zeros_like(depths)
#
#     depths_normalized = depths_normalized.unsqueeze(-1)  # Shape from [N] to [N, 1]
#     depths_normalized = torch.clamp(depths_normalized, 0.0, 1.0)  # 确保在[0,1]范围内
#
#     # Create dummy screenspace points
#     screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)
#
#     # Rasterize using depth values as "colors"
#     rendered_depth, radii = rasterizer(
#         means3D=means3D,
#         means2D=screenspace_points,
#         shs=None,  # No spherical harmonics for depth rendering
#         colors_precomp=depths_normalized,  # Use normalized depth values as "color"
#         opacities=opacity,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp)
#
#     # 确保在返回前所有张量都在CPU上处理
#     depth_min_val = depth_min.cpu().item() if depth_min.numel() > 0 else 0.0
#     depth_max_val = depth_max.cpu().item() if depth_max.numel() > 0 else 1.0
#
#     return {"render": rendered_depth,
#             "viewspace_points": screenspace_points,
#             "visibility_filter": radii > 0,
#             "radii": radii,
#             "depth_range": (depth_min_val, depth_max_val)}  # 使用CPU上的值

def render_depth(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """
    Render depth map from the scene.
    """
    # 获取设备信息
    device = bg_color.device

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取模型参数
    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # 处理协方差计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 计算深度值 - 更安全的方式
    with torch.no_grad():
        # 转换到相机空间
        points_homogeneous = torch.cat([means3D, torch.ones_like(means3D[:, :1], device=device)], dim=-1)
        points_camera = torch.matmul(points_homogeneous, viewpoint_camera.world_view_transform.T)

        # 计算深度值
        depths = points_camera[:, 2] / points_camera[:, 3]

        # 处理异常值
        valid_mask = torch.isfinite(depths)
        if not torch.any(valid_mask):
            # 如果没有有效深度值，使用默认值
            depths_normalized = torch.zeros_like(depths)
        else:
            # 只使用有效值计算范围
            valid_depths = depths[valid_mask]
            depth_min = torch.quantile(valid_depths, 0.02)
            depth_max = torch.quantile(valid_depths, 0.98)

            if depth_max - depth_min > 1e-6:
                depths_normalized = (depths - depth_min) / (depth_max - depth_min)
            else:
                depths_normalized = torch.zeros_like(depths)

            depths_normalized = torch.clamp(depths_normalized, 0.0, 1.0)

        depths_normalized = depths_normalized.unsqueeze(-1)  # [N, 1]

    # 创建屏幕空间点（不需要梯度）
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)

    # 光栅化
    rendered_depth, radii = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=depths_normalized,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # 安全地获取深度范围信息
    depth_min_val = 0.0
    depth_max_val = 1.0
    if torch.any(valid_mask):
        valid_depths = depths[valid_mask]
        depth_min_val = torch.quantile(valid_depths, 0.02).item()
        depth_max_val = torch.quantile(valid_depths, 0.98).item()

    return {
        "render": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_range": (depth_min_val, depth_max_val)
    }


# Export both functions
__all__ = ['render', 'render_depth']