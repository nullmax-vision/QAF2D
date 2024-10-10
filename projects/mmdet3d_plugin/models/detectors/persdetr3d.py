# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations, normlize_boxes
from projects.mmdet3d_plugin.models.utils.proposals_generation import init_proposal_anchors, check_3d
from projects.mmdet3d_plugin.models.utils.checkviews import check, normalize
from projects.mmdet3d_plugin.models.utils.prompt import PadPrompter

import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from torch.nn import functional as F
from thop import profile


@DETECTORS.register_module()
class PersDetr3D(MVXTwoStageDetector):
    """PersDetr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None):
        super(PersDetr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                         pts_middle_encoder, pts_fusion_layer,
                                         img_backbone, pts_backbone, img_neck, pts_neck,
                                         pts_bbox_head, img_roi_head, img_rpn_head,
                                         train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only

        if False:
            self.prompter_0 = PadPrompter(c=256, w=100, h=40)
            self.prompter_1 = PadPrompter(c=256, w=50, h=20)
            self.prompter_2 = PadPrompter(c=256, w=25, h=10)
            self.prompter_3 = PadPrompter(c=256, w=13, h=5)

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            if False:
                flops, params = profile(
                    self.img_backbone, inputs=(img,))
                print('FLOPs = ' + str(flops/(B*1000**3)) + 'G',
                      'Params = ' + str(params/(B*1000**2)) + 'M')

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []

        if self.training and training_mode:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(
                    B, len_queue, int(BN/B / len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)
        else:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(
                    B, int(BN/B/len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                              gt_bboxes_3d=None,
                              gt_labels_3d=None,
                              gt_bboxes=None,
                              gt_labels=None,
                              img_metas=None,
                              centers2d=None,
                              depths=None,
                              gt_bboxes_ignore=None,
                              init_proposals=None,
                              **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key == 'img_feats':
                    data_t[key] = [feat[:, i] for feat in data[key]]
                else:
                    data_t[key] = data[key][:, i]

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                          gt_labels_3d[i], gt_bboxes[i],
                                          gt_labels[i], img_metas[i], centers2d[i], depths[i],
                                          requires_grad=requires_grad, return_losses=return_losses,
                                          init_proposals=init_proposals, **data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses

    def forward_roi_head(self, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes': None}
        else:
            outs_roi = self.img_roi_head(**data)
            return outs_roi

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          init_proposals=None,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(
                    img_metas, init_proposals=init_proposals, **data)
            self.train()
        else:
            outs_roi = self.forward_roi_head(**data)
            outs = self.pts_bbox_head(
                img_metas, init_proposals=init_proposals, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels,
                                 centers2d, outs_roi, depths, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d)

            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def obtain_proposals(self, img, img_metas, training_mode=True):
        self.eval()
        detection_h, detection_w = 736, 1280
        camera_h, camera_w = 900, 1600
        bs, num_cameras, channels, hs, ws = img.shape
        batch_results_proposals = []
        for bs in range(len(img_metas)):
            results_proposals = []
            cam_intrinsics = img_metas[bs]['intrinsics'].data
            cameras_intrinsics = img_metas[bs]['proposals_intrinsics']
            cam_results = img_metas[bs]['results_plane']
            cam_lidar2cams = img_metas[bs]['extrinsics'].data
            for cam_idx in range(img.size(1)):
                if False:
                    img_path = img_metas[bs]['filename'][cam_idx]
                    img_name = img_path.split('/')[-1]
                    cam_image = Image.open(img_path)
                    resized_image = transforms.Resize(
                        size=(camera_h, camera_w))(cam_image)
                    cam_sample = normalize(resized_image)
                    cam_sample = cam_sample[None].to(img.device)

                cam_intrinsic = cameras_intrinsics[cam_idx]
                cam_results_2d = cam_results[cam_idx]
                cam_cam2lidar = np.linalg.inv(cam_lidar2cams[cam_idx])
                results_proposals_cam = []
                # bbox2ds = []
                for ob_idx, result_2d in enumerate(cam_results_2d):
                    bbox2d, class_idx, score, _ = result_2d
                    bbox2d = torch.cat(
                        [torch.tensor([ob_idx]), torch.from_numpy(bbox2d)], dim=0).to(img.device)

                    # 2d results on (900, 1600)
                    scale_factors = torch.zeros_like(bbox2d[1:])
                    scale_factors[0::2] = camera_w / detection_w
                    scale_factors[1::2] = camera_h / detection_h
                    bbox2d[1:] = scale_factors * bbox2d[1:]
                    class_idx = torch.tensor(class_idx).to(img.device)

                    init_proposals_corners, proposals_bboxs, object_class, ious = init_proposal_anchors(
                        class_idx, bbox2d[1:], cam_intrinsic, cam_cam2lidar)
                    if ious.size(0) >= 1:
                        # nms for redunant 3d proposals
                        _, max_ious_idx = torch.topk(
                            ious, k=min(3, ious.size(0)), dim=0, largest=True)
                        proposals_bboxs = proposals_bboxs[max_ious_idx]
                        if False:  # check
                            initial_proposals = init_proposals_corners[max_ious_idx]
                            check_centers = initial_proposals.mean(1)
                            proposals_bboxs = proposals_bboxs
                            centers_embed = proposals_bboxs[..., :3]
                            sizes_embed = proposals_bboxs[..., 3:6]
                            thetas_ = proposals_bboxs[..., -1:]
                            for s, t, c, pt_3d in zip(sizes_embed, thetas_, centers_embed, initial_proposals):
                                w, h, l = s
                                corner_3d = check_3d(c, w, h, l, t)
                                print(1)
                        proposals_bboxs = normlize_boxes(
                            proposals_bboxs, cam_cam2lidar)
                        results_proposals_cam += [(bbox2d, object_class, p)
                                                  for p in proposals_bboxs]
                    # bbox2ds.append(bbox2d[1:])
                results_proposals.append(results_proposals_cam)
            batch_results_proposals.append(results_proposals)
        if training_mode:
            self.train()
        return batch_results_proposals

    def roi_pooling(self, feats, rois, original_size, size=1):
        cf, hf, wf = feats.size()
        assert rois.size(1) == 5
        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)
        norm_oris = rois[:, 1:] / torch.tensor([original_size[0], original_size[1],
                                               original_size[0], original_size[1]])[None].repeat(num_rois, 1).to(feats.device)
        rois = torch.cat(
            [rois[:, :1], norm_oris * torch.tensor([wf, hf, wf, hf])[None].repeat(num_rois, 1).to(feats.device)], dim=1)
        rois = rois.long()
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = feats[..., roi[2].clamp(min=0):(
                roi[4].clamp(max=hf) + 1), roi[1].clamp(min=0):(roi[3].clamp(max=wf) + 1)]
            output.append(F.adaptive_max_pool2d(im, size))

        output = torch.stack(output, 0)
        return output.squeeze(-1).squeeze(-1)

    def extract_roi_feats(self, img_feats, init_proposals):
        level_roi_embed = []

        for num_level in range(len(img_feats)):
            level_feats = img_feats[num_level]
            batch_roi_embed = []
            for b in range(level_feats.size(0)):
                cam_roi_embed = []
                for cam in range(level_feats.size(1)):
                    cam_proposal_b = [bbox2ds[0]
                                      for bbox2ds in init_proposals[b][cam]]
                    if len(cam_proposal_b) > 0:
                        cam_proposal_b = torch.stack(cam_proposal_b, dim=0)
                        cam_roi_embed += [self.roi_pooling(
                            level_feats[b, cam], cam_proposal_b, (1600, 900))]   # (1280, 736)
                    else:
                        cam_roi_embed += []
                batch_roi_embed += [cam_roi_embed]
            level_roi_embed += batch_roi_embed  # batch, cam, objects, dim

        return level_roi_embed

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.extract_feat
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]

        # ---------- initial proposals generation ------------------
        if True:
            initial_proposals = [self.obtain_proposals(
                img=rec_img[:, i, ...], img_metas=img_metas[i])
                for i in range(rec_img.size(1))]  # len_temp 为 len(initial_proposals)
        # ----------------------------------------------------------

        rec_img_feats = self.extract_feat(
            rec_img, self.num_frame_backbone_grads, True)

        # rec_img_feats = [
        #     self.prompter_0(rec_img_feats_[0]),
        #     self.prompter_1(rec_img_feats_[1]),
        #     self.prompter_2(rec_img_feats_[2]),
        #     self.prompter_3(rec_img_feats_[3])
        # ]

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(
                    prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = [torch.cat(
                [prev_img_feats[i], rec_img_feats[i]], dim=1) for i in range(len(self.position_level))]
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                                            gt_labels_3d, gt_bboxes,
                                            gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data,
                                            init_proposals=initial_proposals)

        return losses

    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, init_proposals=None, **data):
        """Test function of point cloud branch."""
        outs_roi = self.forward_roi_head(**data)

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(
            img_metas, init_proposals=init_proposals, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""

        rec_img = data['img'].unsqueeze(1)  # 设置伪时序数据用于检测
        img_metas_temp = [img_metas]
        # ---------- initial proposals generation ------------------
        if True:
            initial_proposals = [self.obtain_proposals(
                img=rec_img[:, i, ...], img_metas=img_metas_temp[i], training_mode=False)
                for i in range(rec_img.size(1))]  # len_temp 为 len(initial_proposals)
        # ----------------------------------------------------------

        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, init_proposals=initial_proposals, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
