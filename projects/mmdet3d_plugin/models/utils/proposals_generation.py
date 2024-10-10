import time
from bdb import effective
from curses import tparm
from locale import normalize
from tkinter import N
from turtle import left
from xmlrpc.client import FastMarshaller
import torch
import numpy as np

Theta = np.arange(24)
Camera_Depth = 1.5*np.arange(2, 69)


def roty(t):
    ''' Rotation about the y-axis. '''
    c = t.cos().cpu().numpy()
    s = t.sin().cpu().numpy()
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def check_3d(c, w, h, l, t):
    x_corners = l / 2 * torch.tensor([1, 1, 1, 1, -1, -1, -1, -1]).to(c.device)
    y_corners = h / 2 * torch.tensor([1, -1, -1, 1, 1, -1, -1, 1]).to(c.device)
    z_corners = w / 2 * torch.tensor([1, 1, -1, -1, 1, 1, -1, -1]).to(c.device)

    R = roty(t[0])

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack(
        [x_corners.cpu(), y_corners.cpu(), z_corners.cpu()]))
    corners_3d = torch.from_numpy(corners_3d).to(c.device)

    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + c[0]
    corners_3d[1, :] = corners_3d[1, :] + c[1]
    corners_3d[2, :] = corners_3d[2, :] + c[2]

    return corners_3d


def multi_view_points(points: torch.tensor, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[1] == 3

    '''
    I, 0
    0, 1
    shape: 4*4 matrix
    '''
    viewpad = torch.eye(4).to(points.device)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[-1]
    num_corners = points.size(0)

    # Do operation in homogenous coordinates.
    points = torch.cat(
        (points, torch.ones(num_corners, 1, nbr_points).to(points.device)), dim=1)
    expanded_viewpad = viewpad[None, ...].repeat(num_corners, 1, 1)
    points = torch.bmm(expanded_viewpad.float(), points.float())
    points = points[:, :3, :]

    if normalize:
        points = points / points[:, 2:3, :].repeat(1, 3, 1)

    if False:
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3,
                                     :].repeat(3, 0).reshape(3, nbr_points)

    return points


def obtain_center_faster(box_side, ds, thetas, box_sizes, viewed_inv, cam2lidar):
    left_pts = box_side
    len_ds = ds.size(0)
    left_pts_d = left_pts.repeat(len_ds, 1, 1) * ds[:, None, None]
    left_pts = torch.cat((left_pts_d, torch.ones_like(left_pts_d)), dim=-1)
    left_pts[..., 2] = left_pts[..., 2] * ds[:, None]
    left_pts = left_pts[..., :3].permute(0, 2, 1)
    viewed_inv = viewed_inv[None].repeat(len_ds, 1, 1)
    left_pts_3d = torch.matmul(viewed_inv, left_pts.float()).permute(0, 2, 1)
    ws, ls = box_sizes[..., 0], box_sizes[..., 1]
    # 旋转偏移量计算
    warp_x_offsets = -1 * ws[:, None] * thetas.sin().view(1, -1)
    warp_z_offsets = -1 * ws[:, None] * thetas.cos().view(1, -1)
    # 计算八角点之间的偏移
    x_offsets = ls[:, None] * thetas.cos().view(1, -1)
    y_offsets = torch.zeros_like(x_offsets)
    z_offsets = -1 * ls[:, None] * thetas.sin().view(1, -1)

    len_thetas = warp_x_offsets.size(1)
    len_sizes = box_sizes.size(0)
    warp_x_offsets = warp_x_offsets.unsqueeze(
        -1).unsqueeze(-1).repeat(1, 1, len_ds, 2)
    warp_z_offsets = warp_z_offsets.unsqueeze(
        -1).unsqueeze(-1).repeat(1, 1, len_ds, 2)
    thetas_ = thetas[None, :, None].repeat(len_sizes, 1, len_ds)
    ws_ = ws[:, None, None, None].repeat(1, len_thetas, len_ds, 1)
    ls_ = ls[:, None, None, None].repeat(1, len_thetas, len_ds, 1)

    x_offsets = x_offsets[..., None, None].repeat(1, 1, len_ds, 4)
    y_offsets = y_offsets[..., None, None].repeat(1, 1, len_ds, 4)
    z_offsets = z_offsets[..., None, None].repeat(1, 1, len_ds, 4)
    # 计算偏转平面
    left_pts_3d = left_pts_3d.unsqueeze(0).unsqueeze(
        0).repeat(len_sizes, len_thetas, 1, 1, 1)
    rside_pts_3d = torch.zeros_like(left_pts_3d)
    rside_pts_3d[..., 0] = left_pts_3d[..., 0] + warp_x_offsets
    rside_pts_3d[..., 1] = left_pts_3d[..., 1]
    rside_pts_3d[..., -1] = left_pts_3d[..., -1] + warp_z_offsets
    pts_3ds = torch.cat((left_pts_3d, rside_pts_3d), dim=3)
    # 计算得出八角点
    offsets = torch.stack([x_offsets, y_offsets, z_offsets], dim=-1)
    pts_3ds_1 = pts_3ds + offsets
    pts_3ds = torch.cat((pts_3ds, pts_3ds_1), dim=3)
    pts_3ds = pts_3ds.view(-1, pts_3ds.size(-2), pts_3ds.size(-1))
    # 计算在lidar坐标系下的八角点
    pts_3d = torch.cat([pts_3ds, torch.ones_like(pts_3ds)],
                       dim=-1)[..., :4].float()
    lidar_pts_3d = pts_3d @ torch.from_numpy(
        cam2lidar).t().to(pts_3d.device)[..., :3]

    # 计算bbox proposals
    if True:  # camera proposals
        centers_embed = pts_3d.mean(1)[..., :3]
        thetas_ = thetas_.view(-1)[:, None]
        heighs = (pts_3ds[:, 0::2, 1] - pts_3ds[:, 1::2, 1]
                  ).abs().mean(1)[:, None]
        ws_ = ws_.view(-1, 1)
        ls_ = ls_.view(-1, 1)
        assert ws_.shape == heighs.shape and ls_.shape == ls_.shape
        # theta_embed = torch.cat((thetas_.sin(), thetas_.cos()), dim=-1)
        theta_embed = thetas_
        sizes_embed = torch.cat((ws_, heighs, ls_), dim=-1)  # .log()
        bboxs_3ds = torch.cat(
            (centers_embed, sizes_embed, theta_embed), dim=-1)
    else:  # lidar proposals
        centers_embed = lidar_pts_3d.mean(1)
        thetas_ = thetas_.view(-1)
        thetas_ = (-1 * thetas_ - np.pi / 2)[:, None]
        heighs = (pts_3ds[:, 0::2, 1] - pts_3ds[:, 1::2, 1]
                  ).abs().mean(1)[:, None]
        ws_ = ws_.view(-1, 1)
        ls_ = ls_.view(-1, 1)
        assert ws_.shape == heighs.shape and ls_.shape == ls_.shape
        theta_embed = torch.cat((thetas_.sin(), thetas_.cos()), dim=-1)
        sizes_embed = torch.cat((ws_, ls_, heighs), dim=-1).log()
        bboxs_3ds = torch.cat(
            (centers_embed[:, :2], sizes_embed[:, :2], centers_embed[:, 2:], sizes_embed[:, 2:], theta_embed), dim=-1)

    return pts_3ds, bboxs_3ds


def obtain_cneter(box_2d, d, theta, box_size, viewed_inv):

    # 以left side为基准
    left_pts = box_2d[0:2]
    left_pts = torch.cat((d * left_pts, torch.ones_like(left_pts)), dim=-1)
    left_pts[:, 2] = d
    left_pts = left_pts[:, :3].t()
    left_pts_3d = torch.matmul(viewed_inv, left_pts.float()).t()

    # 根据角度，计算偏移平面
    w, h, l = box_size
    rside_pts_3d = torch.zeros_like(left_pts_3d)
    warp_x_offset = -1 * w * torch.sin(torch.tensor(theta))
    warp_z_offset = -1 * w * torch.cos(torch.tensor(theta))

    rside_pts_3d[:, 0] = left_pts_3d[:, 0] + warp_x_offset
    rside_pts_3d[:, 1] = left_pts_3d[:, 1]
    rside_pts_3d[:, -1] = left_pts_3d[:, -1] + warp_z_offset

    pts_3d = torch.cat((left_pts_3d, rside_pts_3d), dim=0)

    # 计算投影的八角点
    x_offset = l * torch.cos(torch.tensor(theta))
    y_offset = torch.zeros_like(torch.tensor(theta))
    z_offset = -1 * l * torch.sin(torch.tensor(theta))

    offset = torch.stack([x_offset, y_offset, z_offset], dim=0).view(1, -1)
    pts_3d_1 = pts_3d + offset
    pts_3d = torch.cat((pts_3d, pts_3d_1), dim=0)

    center_1 = pts_3d[:4].mean(0)
    center_2 = pts_3d[4:].mean(0)
    center_embed = (center_1 + center_2) / 2
    heigh = (pts_3d[0::2, 1] - pts_3d[1::2, 1]).abs().mean()
    sizes_embed = torch.tensor([w, heigh, l])
    theta_embed = torch.tensor(
        [torch.tensor(theta).sin(), torch.tensor(theta).cos()])
    bboxs_3d = torch.cat(
        (center_embed[:2], sizes_embed[:2], center_embed[2:], sizes_embed[2:], theta_embed), dim=0)

    return [pts_3d], [bboxs_3d]


def calculate_ious(proposals_bbox2ds, gt_bbox2d):
    x1, y1, x2, y2 = gt_bbox2d
    pxs1, pys1, pxs2, pys2 = proposals_bbox2ds

    # 计算iou
    areas = (x2-x1)*(y2-y1)
    areas = areas[None].expand_as(pxs1)

    proposals_areas = (pxs2-pxs1)*(pys2-pys1)

    inter_x1 = torch.max(torch.cartesian_prod(
        x1[None].float(), pxs1.float()), dim=-1)[0]
    inter_y1 = torch.max(torch.cartesian_prod(
        y1[None].float(), pys1.float()), dim=-1)[0]
    inter_x2 = torch.min(torch.cartesian_prod(
        x2[None].float(), pxs2.float()), dim=-1)[0]
    inter_y2 = torch.min(torch.cartesian_prod(
        y2[None].float(), pys2.float()), dim=-1)[0]

    inter_areas = (inter_x2-inter_x1).clamp(min=0.0) * \
        (inter_y2-inter_y1).clamp(min=0.0)
    ious = inter_areas / (areas + proposals_areas - inter_areas)

    return ious


def init_proposal_anchors(cls, bbox2d, camera_intrinsic, cam2lidar):

    if cls == 3:   # 'Car'
        obj_w = torch.from_numpy(np.arange(14.5, 27.5 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(12.5, 31.0 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(34.5, 65.5 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 0:   # 'Pedestrian'
        obj_w = torch.from_numpy(np.arange(3.4, 10.2 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(11.5, 22.0 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(3.3, 12.8 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 4:    # 'Bus'
        obj_w = torch.from_numpy(np.arange(25.5, 35 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(28, 45.5 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(69, 138.1 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 5:    # 'Truck'
        obj_w = torch.from_numpy(np.arange(17, 30 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(17, 40 + 5, 5)/10.)
        # obj_l = torch.from_numpy(np.arange(45, 103 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(17.8, 140 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 'Trailer':  # 半挂车， 需要重新计算
        obj_w = torch.from_numpy(np.arange(21.6, 23.3 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(33.1, 38.8 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(17.8, 140 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 7:  # 'Construction_vehicle'
        obj_w = torch.from_numpy(np.arange(21.1, 33.4 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(20.3, 29.3 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(37.1, 75.5 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 2:   # 'Motorcycle'
        obj_w = torch.from_numpy(np.arange(4.7, 14.1 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(11.8, 19.3 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(12.5, 27.5 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 1:   # 'Bicycle'
        obj_w = torch.from_numpy(np.arange(4.3, 9.3 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(9.5, 19.3 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(13.3, 20 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 8:   # 'Traffic_cone'
        obj_w = torch.from_numpy(np.arange(2.1, 12.1 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(5.1, 13.7 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(3, 7.3 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 6:    # 'Barrier'
        obj_w = torch.from_numpy(np.arange(17.3, 35.8 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(8.8, 13.8 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(3.9, 7.8 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    elif cls == 9:    # 'bicycle_rack'
        obj_w = torch.from_numpy(np.arange(21.6, 23.3 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(17.8, 14.0 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(33.1, 38.9 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)
    else:
        obj_w = torch.from_numpy(np.arange(2.1, 35.9 + 5, 5)/10.)
        obj_h = torch.from_numpy(np.arange(5.1, 45.9 + 5, 5)/10.)
        obj_l = torch.from_numpy(np.arange(3.0, 140 + 5, 5)/10.)
        obj_size = torch.cartesian_prod(obj_w, obj_l)

    # 2. obtain corner points
    x_min, y_min, x_max, y_max = bbox2d
    x_corners = torch.stack((x_min, x_min, x_max, x_max), dim=0)
    y_corners = torch.stack((y_min, y_max, y_min, y_max), dim=0)
    corners = torch.stack((x_corners, y_corners), dim=-1)

    # 3. obtain camera project matrix
    projected_matrix = torch.from_numpy(camera_intrinsic).to(corners.device)
    projected_matrix = projected_matrix[:3, :3]
    projected_matrix_inv = torch.inverse(projected_matrix)
    viewed_inv = projected_matrix_inv.float()

    init_proposals = list()
    init_bbox3ds = list()
    init_classes = list()

    o_sizes = obj_size.to(corners.device)
    ds = torch.from_numpy(Camera_Depth).to(corners.device)
    thetas = (Theta / 12.) * np.pi  # - (1/2)*np.pi
    thetas = torch.from_numpy(thetas).to(corners.device)
    corners_3ds_left, bboxs_3ds_left = obtain_center_faster(
        corners[0:2], ds, thetas, o_sizes, viewed_inv, cam2lidar)
    corners_3ds_right, bboxs_3ds_right = obtain_center_faster(
        corners[2:], ds, thetas, o_sizes, viewed_inv, cam2lidar)

    corners_3ds = torch.cat([corners_3ds_left, corners_3ds_right], dim=0)
    bboxs_3ds = torch.cat([bboxs_3ds_left, bboxs_3ds_right], dim=0)

    hs = (corners_3ds[..., 0::2, 1] - corners_3ds[..., 1::2, 1]).abs().mean(1)
    hs_flag = torch.nonzero(torch.eq(hs >= obj_h.min(), hs <= obj_h.max()))
    ious_ = []
    if hs_flag.size(0) > 0:
        hs_flag = hs_flag.view(-1)
        corners_3ds = corners_3ds[hs_flag]
        bboxs_3ds = bboxs_3ds[hs_flag]
        img_corners = multi_view_points(
            points=corners_3ds.permute(0, 2, 1), view=projected_matrix, normalize=True)
        img_corners = img_corners.permute(0, 2, 1)[..., :2]
        corners_2d_x = img_corners[..., 0]
        corners_2d_y = img_corners[..., 1]
        pxs_min, pxs_max = corners_2d_x.min(-1)[0], corners_2d_x.max(-1)[0]
        pys_min, pys_max = corners_2d_y.min(-1)[0], corners_2d_y.max(-1)[0]
        ious = calculate_ious(
            proposals_bbox2ds=(pxs_min, pys_min, pxs_max, pys_max), gt_bbox2d=(
                x_min, y_min, x_max, y_max))
        e_idx = torch.nonzero(ious > 0.98).view(-1)
        if e_idx.size(0) > 0:
            init_bbox3d = bboxs_3ds[e_idx]
            init_proposal = corners_3ds[e_idx]
            init_proposals += [p for p in init_proposal]
            init_bbox3ds += [b for b in init_bbox3d]
            init_proposals = torch.stack(init_proposals, dim=0)
            init_bbox3ds = torch.stack(init_bbox3ds, dim=0)
            ious_ = ious[e_idx]
    ious_ = torch.tensor(ious_)

    return init_proposals, init_bbox3ds, cls, ious_
