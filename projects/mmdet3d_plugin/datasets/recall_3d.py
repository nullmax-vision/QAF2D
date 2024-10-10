import logging
from typing import Final, Tuple

import numpy as np
import pandas as pd
from enum import Enum

# import refile
import numpy as np
# import pyarrow.feather as feather
# import av2.utils.io as io_utils
import logging
import warnings
from multiprocessing import get_context
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Union, Final, Any
# from joblib import Parallel, delayed
from pathlib import Path
# from upath import UPath
from io import BytesIO

from dataclasses import dataclass
import json

from av2.evaluation.detection.utils import DetectionCfg
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt
from av2.utils.constants import EPS
from av2.evaluation.detection.constants import (
    InterpType,
)
from av2.evaluation.detection.utils import (
    compute_average_precision,
    groupby,
)
from joblib import Parallel, delayed
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.io import TimestampedCitySE3EgoPoses
from av2.utils.typing import NDArrayBool, NDArrayFloat
import av2.geometry.geometry as geometry_utils
from av2.utils.io import TimestampedCitySE3EgoPoses, read_city_SE3_ego

from av2.map.drivable_area import DrivableArea
from av2.map.lane_segment import LaneSegment
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.map.map_api import DrivableAreaMapLayer, RoiMapLayer, GroundHeightLayer
from av2.geometry.sim2 import Sim2
import av2.utils.io as io_utils

from av2.utils.io import TimestampedCitySE3EgoPoses, read_city_SE3_ego
from av2.evaluation.detection.constants import NUM_DECIMALS, TruePositiveErrorNames

from scipy.spatial.distance import cdist

from av2.evaluation.detection.constants import (
    MAX_NORMALIZED_ASE,
    MAX_SCALE_ERROR,
    MAX_YAW_RAD_ERROR,
    MIN_AP,
    MIN_CDS,
    AffinityType,
    CompetitionCategories,
    DistanceType,
    FilterMetricType,
)
from av2.geometry.geometry import mat_to_xyz, quat_to_mat, wrap_angles
from av2.geometry.iou import iou_3d_axis_aligned
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt
import kornia.geometry.conversions as C
import torch
from torch import Tensor
from math import pi as PI

TP_ERROR_COLUMNS: Final[Tuple[str, ...]] = tuple(x.value for x in TruePositiveErrorNames)
DTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("num_interior_pts",)
UUID_COLUMN_NAMES: Final[Tuple[str, ...]] = (
    "log_id",
    "timestamp_ns",
    "category",
)

class TruePositiveErrorNames(str, Enum):
    """True positive error names."""

    ATE = "ATE"
    ASE = "ASE"
    AOE = "AOE"


class MetricNames(str, Enum):
    """Metric names."""

    AP = "AP"
    ATE = TruePositiveErrorNames.ATE.value
    ASE = TruePositiveErrorNames.ASE.value
    AOE = TruePositiveErrorNames.AOE.value
    CDS = "CDS"
    RECALL = "RECALL"

class GroundHeightLayerRemote(GroundHeightLayer):
    @classmethod
    def from_file_remote(cls, log_map_dirpath: Path, log_map_dirpath_s3) -> GroundHeightLayer:
        """Load ground height values (w/ values at 30 cm resolution) from .npy file, and associated Sim(2) mapping.
        Note: ground height values are stored on disk as a float16 2d-array, but cast to float32 once loaded for
        compatibility with matplotlib.
        Args:
            log_map_dirpath: path to directory which contains map files associated with one specific log/scenario.
        Returns:
            The ground height map layer.
        Raises:
            RuntimeError: If raster ground height layer file is missing or Sim(2) mapping from city to image coordinates
                is missing.
        """
        ground_height_npy_fpaths = sorted(log_map_dirpath.glob("*_ground_height_surface____*.npy"))
        # ground_height_npy_fpaths = sorted(refile.smart_glob(log_map_dirpath_s3 + "/*_ground_height_surface____*.npy"))
        if not len(ground_height_npy_fpaths) == 1:
            raise RuntimeError("Raster ground height layer file is missing")

        Sim2_json_fpaths = sorted(log_map_dirpath.glob("*___img_Sim2_city.json"))
        # Sim2_json_fpaths = sorted(refile.smart_glob(log_map_dirpath_s3 + "/*___img_Sim2_city.json"))
        if not len(Sim2_json_fpaths) == 1:
            raise RuntimeError("Sim(2) mapping from city to image coordinates is missing")

        # load the file with rasterized values
        with ground_height_npy_fpaths[0].open("rb") as f:
        # with refile.smart_open(ground_height_npy_fpaths[0], "rb") as f:
            _bytes = f.read()
        ground_height_array: NDArrayFloat = np.load(BytesIO(_bytes))

        array_Sim2_city = Sim2.from_json(Sim2_json_fpaths[0])
        # with refile.smart_open(Sim2_json_fpaths[0], "r") as f:
        #     json_data = json.load(f)
        # R: NDArrayFloat = np.array(json_data["R"]).reshape(2, 2)
        # t: NDArrayFloat = np.array(json_data["t"]).reshape(2)
        # s = float(json_data["s"])
        # array_Sim2_city = Sim2(R, t, s)

        return cls(array=ground_height_array.astype(float), array_Sim2_city=array_Sim2_city)

class ArgoverseStaticMapRemote(ArgoverseStaticMap):
    """API to interact with a local map for a single log (within a single city).
    """

    @classmethod
    def from_json_remote(cls, static_map_path_s3, static_map_path: Path) -> ArgoverseStaticMap:
        """Instantiate an Argoverse static map object (without raster data) from a JSON file containing map data.
        Args:
            static_map_path: Path to the JSON file containing map data. The file name must match
                the following pattern: "log_map_archive_{log_id}.json".
        Returns:
            An Argoverse HD map.
        """
        vector_data = io_utils.read_json_file(static_map_path)
        log_id = static_map_path.stem.split("log_map_archive_")[1]
        # with refile.smart_open(static_map_path_s3, "rb") as f:
        #     vector_data: Dict[str, Any] = json.load(f)

        vector_drivable_areas = {da["id"]: DrivableArea.from_dict(da) for da in vector_data["drivable_areas"].values()}
        vector_lane_segments = {ls["id"]: LaneSegment.from_dict(ls) for ls in vector_data["lane_segments"].values()}

        if "pedestrian_crossings" not in vector_data:
            logger.error("Missing Pedestrian crossings!")
            vector_pedestrian_crossings = {}
        else:
            vector_pedestrian_crossings = {
                pc["id"]: PedestrianCrossing.from_dict(pc) for pc in vector_data["pedestrian_crossings"].values()
            }

        return cls(
            log_id=log_id,
            vector_drivable_areas=vector_drivable_areas,
            vector_lane_segments=vector_lane_segments,
            vector_pedestrian_crossings=vector_pedestrian_crossings,
            raster_drivable_area_layer=None,
            raster_roi_layer=None,
            raster_ground_height_layer=None,
        )

    @classmethod
    def from_map_dir_remote(cls, log_map_dirpath: Path, build_raster: bool = False, data_root='') -> ArgoverseStaticMap:
        """Instantiate an Argoverse map object from data stored within a map data directory.
        Note: the ground height surface file and associated coordinate mapping is not provided for the
        2.0 Motion Forecasting dataset, so `build_raster` defaults to False. If raster functionality is
        desired, users should pass `build_raster` to True (e.g. for the Sensor Datasets and Map Change Datasets).
        Args:
            log_map_dirpath: Path to directory containing scenario-specific map data,
                JSON file must follow this schema: "log_map_archive_{log_id}.json".
            build_raster: Whether to rasterize drivable areas, compute region of interest BEV binary segmentation,
                and to load raster ground height from disk (when available).
        Returns:
            The HD map.
        Raises:
            RuntimeError: If the vector map data JSON file is missing.
        """
        # log_map_dirpath_s3 = str(log_map_dirpath).replace(data_root, 's3://argo/argo_data/')
        # vector_data_fnames = sorted(refile.smart_glob(log_map_dirpath + "/log_map_archive_*.json"))
        vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        # "s3://argo/argo_data/val/02678d04-cc9f-3148-9f95-1ba66347dff9/map/log_map_archive_*.json"
        # Load vector map data from JSON file
        # vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
        vector_data_fname = vector_data_fnames[0]

        vector_data_json_path = vector_data_fname

        static_map = cls.from_json_remote(vector_data_json_path, Path(vector_data_json_path))
        static_map.log_id = log_map_dirpath.parent.stem

        # Avoid file I/O and polygon rasterization when not needed
        if build_raster:
            drivable_areas: List[DrivableArea] = list(static_map.vector_drivable_areas.values())
            static_map.raster_drivable_area_layer = DrivableAreaMapLayer.from_vector_data(drivable_areas=drivable_areas)
            static_map.raster_roi_layer = RoiMapLayer.from_drivable_area_layer(static_map.raster_drivable_area_layer)
            static_map.raster_ground_height_layer = GroundHeightLayerRemote.from_file_remote(log_map_dirpath, log_map_dirpath)
            # static_map.raster_ground_height_layer = GroundHeightLayerRemote.from_file_remote(log_map_dirpath, log_map_dirpath_s3)

        return static_map


logger = logging.getLogger(__name__)

def load_mapped_avm_and_egoposes(log_ids: List[str], dataset_dir: Path
    ) -> Tuple[Dict[str, ArgoverseStaticMap], Dict[str, TimestampedCitySE3EgoPoses]]:
        """Load the maps and egoposes for each log in the dataset directory.
        Args:
            log_ids: List of the log_ids.
            dataset_dir: Directory to the dataset.
        Returns:
            A tuple of mappings from log id to maps and timestamped-egoposes, respectively.
        Raises:
            RuntimeError: If the process for loading maps and timestamped egoposes fails.
        """
        log_id_to_timestamped_poses = {log_id: read_city_SE3_ego(dataset_dir / log_id) for log_id in log_ids}
        avms: Optional[List[ArgoverseStaticMap]] = Parallel(n_jobs=-1, backend="threading")(
            delayed(ArgoverseStaticMapRemote.from_map_dir_remote)(dataset_dir / log_id / "map", build_raster=True) for log_id in log_ids)

        if avms is None:
            raise RuntimeError("Map and egopose loading has failed!")
        log_id_to_avm = {log_ids[i]: avm for i, avm in enumerate(avms)}
        return log_id_to_avm, log_id_to_timestamped_poses

def compute_objects_in_roi_mask(cuboids_ego: NDArrayFloat, city_SE3_ego: SE3, avm: ArgoverseStaticMap) -> NDArrayBool:

    is_within_roi: NDArrayBool
    if len(cuboids_ego) == 0:
        is_within_roi = np.zeros((0,), dtype=bool)
        return is_within_roi
    cuboid_list_ego: CuboidList = CuboidList([Cuboid.from_numpy(params) for params in cuboids_ego])
    cuboid_list_city = cuboid_list_ego.transform(city_SE3_ego)
    cuboid_list_vertices_m_city = cuboid_list_city.vertices_m

    is_within_roi = avm.get_raster_layer_points_boolean(
        cuboid_list_vertices_m_city.reshape(-1, 3)[..., :2], RasterLayerType.ROI
    )
    is_within_roi = is_within_roi.reshape(-1, 8)
    is_within_roi = is_within_roi.any(axis=1)
    return is_within_roi

def compute_evaluated_dts_mask(
    xyz_m_ego: NDArrayFloat,
    cfg: DetectionCfg,
) -> NDArrayBool:

    is_evaluated: NDArrayBool
    if len(xyz_m_ego) == 0:
        is_evaluated = np.zeros((0,), dtype=bool)
        return is_evaluated
    norm: NDArrayFloat = np.linalg.norm(xyz_m_ego, axis=1)  # type: ignore
    # is_evaluated = norm < cfg.max_range_m
    is_evaluated = np.logical_and(norm > cfg.eval_range_m[0], norm < cfg.eval_range_m[1])

    cumsum: NDArrayInt = np.cumsum(is_evaluated)
    max_idx_arr: NDArrayInt = np.where(cumsum > cfg.max_num_dts_per_category)[0]
    if len(max_idx_arr) > 0:
        max_idx = max_idx_arr[0]
        is_evaluated[max_idx:] = False  # type: ignore
    return is_evaluated

def compute_evaluated_gts_mask(
    xyz_m_ego: NDArrayFloat,
    num_interior_pts: NDArrayInt,
    cfg: DetectionCfg,
) -> NDArrayBool:

    is_evaluated: NDArrayBool
    if len(xyz_m_ego) == 0:
        is_evaluated = np.zeros((0,), dtype=bool)
        return is_evaluated
    norm: NDArrayFloat = np.linalg.norm(xyz_m_ego, axis=1)  # type: ignore
    is_evaluated_range = np.logical_and(norm > cfg.eval_range_m[0], norm < cfg.eval_range_m[1])
    is_evaluated = np.logical_and(is_evaluated_range, num_interior_pts > 0)
    # is_evaluated = np.logical_and(norm < cfg.eval_range_m[1], num_interior_pts > 0)
    
    return is_evaluated

def distance(dts: NDArrayFloat, gts: NDArrayFloat, metric: DistanceType) -> NDArrayFloat:

    if metric == DistanceType.TRANSLATION:
        translation_errors: NDArrayFloat = np.linalg.norm(dts - gts, axis=1)  # type: ignore
        return translation_errors
    elif metric == DistanceType.SCALE:
        scale_errors: NDArrayFloat = 1 - iou_3d_axis_aligned(dts, gts)
        return scale_errors
    elif metric == DistanceType.ORIENTATION:
        yaws_dts: NDArrayFloat = mat_to_xyz(quat_to_mat(dts))[..., 2]
        yaws_gts: NDArrayFloat = mat_to_xyz(quat_to_mat(gts))[..., 2]
        orientation_errors = wrap_angles(yaws_dts - yaws_gts)
        return orientation_errors
    else:
        raise NotImplementedError("This distance metric is not implemented!")
    
def compute_affinity_matrix(dts: NDArrayFloat, gts: NDArrayFloat, metric: AffinityType) -> NDArrayFloat:

    if metric == AffinityType.CENTER:
        dts_xy_m = dts
        gts_xy_m = gts
        affinities: NDArrayFloat = -cdist(dts_xy_m, gts_xy_m)
    else:
        raise NotImplementedError("This affinity metric is not implemented!")
    return affinities

def assign(dts: NDArrayFloat, gts: NDArrayFloat, cfg: DetectionCfg) -> Tuple[NDArrayFloat, NDArrayFloat]:

    affinity_matrix = compute_affinity_matrix(dts[..., :3], gts[..., :3], cfg.affinity_type)

    # Get the GT label for each max-affinity GT label, detection pair.
    idx_gts = affinity_matrix.argmax(axis=1)[None]

    # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
    # We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
    # The following line grabs the max affinity for each detection to a ground truth label.
    affinities: NDArrayFloat = np.take_along_axis(affinity_matrix.transpose(), idx_gts, axis=0)[0]  # type: ignore

    # Find the indices of the _first_ detection assigned to each GT.
    assignments: Tuple[NDArrayInt, NDArrayInt] = np.unique(idx_gts, return_index=True)  # type: ignore
    idx_gts, idx_dts = assignments

    T, E = len(cfg.affinity_thresholds_m), 3
    dts_metrics: NDArrayFloat = np.zeros((len(dts), T + E))
    dts_metrics[:, T:] = cfg.metrics_defaults[1:4]
    gts_metrics: NDArrayFloat = np.zeros((len(gts), T + E))
    gts_metrics[:, T:] = cfg.metrics_defaults[1:4]
    for i, threshold_m in enumerate(cfg.affinity_thresholds_m):
        is_tp: NDArrayBool = affinities[idx_dts] > -threshold_m

        dts_metrics[idx_dts[is_tp], i] = True
        gts_metrics[idx_gts, i] = True

        if threshold_m != cfg.tp_threshold_m:
            continue  # Skip if threshold isn't the true positive threshold.
        if not np.any(is_tp):
            continue  # Skip if no true positives exist.

        idx_tps_dts: NDArrayInt = idx_dts[is_tp]
        idx_tps_gts: NDArrayInt = idx_gts[is_tp]

        tps_dts = dts[idx_tps_dts]
        tps_gts = gts[idx_tps_gts]

        translation_errors = distance(tps_dts[:, :3], tps_gts[:, :3], DistanceType.TRANSLATION)
        scale_errors = distance(tps_dts[:, 3:6], tps_gts[:, 3:6], DistanceType.SCALE)
        orientation_errors = distance(tps_dts[:, 6:10], tps_gts[:, 6:10], DistanceType.ORIENTATION)
        dts_metrics[idx_tps_dts, T:] = np.stack((translation_errors, scale_errors, orientation_errors), axis=-1)
    return dts_metrics, gts_metrics


def accumulate(
    dts: NDArrayFloat,
    gts: NDArrayFloat,
    cfg: DetectionCfg,
    avm: Optional[ArgoverseStaticMap] = None,
    city_SE3_ego: Optional[SE3] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    
    N, M = len(dts), len(gts)
    T, E = len(cfg.affinity_thresholds_m), 3

    # Sort the detections by score in _descending_ order.
    scores: NDArrayFloat = dts[..., -1]
    permutation: NDArrayInt = np.argsort(-scores).tolist()
    dts = dts[permutation]

    is_evaluated_dts: NDArrayBool = np.ones(N, dtype=bool)
    is_evaluated_gts: NDArrayBool = np.ones(M, dtype=bool)
    if avm is not None and city_SE3_ego is not None:
        is_evaluated_dts &= compute_objects_in_roi_mask(dts, city_SE3_ego, avm)
        is_evaluated_gts &= compute_objects_in_roi_mask(gts, city_SE3_ego, avm)

    is_evaluated_dts &= compute_evaluated_dts_mask(dts[..., :3], cfg)
    is_evaluated_gts &= compute_evaluated_gts_mask(gts[..., :3], gts[..., -1], cfg)

    # Initialize results array.
    dts_augmented: NDArrayFloat = np.zeros((N, T + E + 1))
    gts_augmented: NDArrayFloat = np.zeros((M, T + E + 1))

    # `is_evaluated` boolean flag is always the last column of the array.
    dts_augmented[is_evaluated_dts, -1] = True
    gts_augmented[is_evaluated_gts, -1] = True

    if is_evaluated_dts.sum() > 0 and is_evaluated_gts.sum() > 0:
        # Compute true positives by assigning detections and ground truths.
        dts_assignments, gts_assignments = assign(dts[is_evaluated_dts], gts[is_evaluated_gts], cfg)
        dts_augmented[is_evaluated_dts, :-1] = dts_assignments
        gts_augmented[is_evaluated_gts, :-1] = gts_assignments

    # Permute the detections according to the original ordering.
    outputs: Tuple[NDArrayInt, NDArrayInt] = np.unique(permutation, return_index=True)  # type: ignore
    _, inverse_permutation = outputs
    dts_augmented = dts_augmented[inverse_permutation]
    return dts_augmented, gts_augmented


def interpolate_precision(precision: NDArrayFloat, interpolation_method: InterpType = InterpType.ALL) -> NDArrayFloat:
    r"""Interpolate the precision at each sampled recall.

    This function smooths the precision-recall curve according to the method introduced in Pascal
    VOC:

    Mathematically written as:
        $$p_{\text{interp}}(r) = \max_{\tilde{r}: \tilde{r} \geq r} p(\tilde{r})$$

    See equation 2 in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
        for more information.

    Args:
        precision: Precision at all recall levels (N,).
        interpolation_method: Accumulation method.

    Returns:
        (N,) The interpolated precision at all sampled recall levels.

    Raises:
        NotImplementedError: If the interpolation method is not implemented.
    """
    precision_interpolated: NDArrayFloat
    if interpolation_method == InterpType.ALL:
        precision_interpolated = np.maximum.accumulate(precision[::-1])[::-1]
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return precision_interpolated

def compute_average_precision(
    tps: NDArrayBool, recall_interpolated: NDArrayFloat, num_gts: int
) -> Tuple[float, NDArrayFloat]:
    """Compute precision and recall, interpolated over N fixed recall points.

    Args:
        tps: True positive detections (ranked by confidence).
        recall_interpolated: Interpolated recall values.
        num_gts: Number of annotations of this class.

    Returns:
        The average precision and interpolated precision values.
    """
    cum_tps: NDArrayInt = np.cumsum(tps)
    cum_fps: NDArrayInt = np.cumsum(~tps)
    cum_fns: NDArrayInt = num_gts - cum_tps

    # Compute precision.
    precision: NDArrayFloat = cum_tps / (cum_tps + cum_fps + EPS)

    # Compute recall.
    recall: NDArrayFloat = cum_tps / (cum_tps + cum_fns)

    # Interpolate precision -- VOC-style.
    precision = interpolate_precision(precision)

    # Evaluate precision at different recalls.
    precision_interpolated: NDArrayFloat = np.interp(recall_interpolated, recall, precision, right=0)  # type: ignore

    average_precision: float = np.mean(precision_interpolated)
    recall3d: float = cum_tps[-1] / num_gts
    return average_precision, precision_interpolated, recall3d

def summarize_metrics(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> pd.DataFrame:
    """Calculate and print the 3D object detection metrics.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(0, 1, cfg.num_recall_samples, endpoint=True)

    # Initialize the summary metrics.
    summary = pd.DataFrame(
        {s.value: cfg.metrics_defaults[i] for i, s in enumerate(tuple(MetricNames))}, index=cfg.categories
    )

    average_precisions = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    average_recall = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    for category in cfg.categories:
        # Find detections that have the current category.
        is_category_dts = dts["category"] == category

        # Only keep detections if they match the category and have NOT been filtered.
        is_valid_dts = np.logical_and(is_category_dts, dts["is_evaluated"])

        # Get valid detections and sort them in descending order.
        category_dts = dts.loc[is_valid_dts].sort_values(by="score", ascending=False).reset_index(drop=True)

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category

        # Compute number of ground truth annotations.
        num_gts = gts.loc[is_category_gts, "is_evaluated"].sum()

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = category_dts[affinity_threshold_m].astype(bool).to_numpy()
            threshold_average_precision, _, recall = compute_average_precision(true_positives, recall_interpolated, num_gts)

            # Record the average precision.
            average_precisions.loc[category, affinity_threshold_m] = threshold_average_precision
            average_recall.loc[category, affinity_threshold_m] = recall

        mean_average_precisions: NDArrayFloat = average_precisions.loc[category].to_numpy().mean()
        mean_average_recall: NDArrayFloat = average_recall.loc[category].to_numpy().mean()

    return mean_average_recall, average_recall

def evaluate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
    n_jobs: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a set of detections against the ground truth annotations.

    Each sweep is processed independently, computing assignment between detections and ground truth annotations.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.
        n_jobs: Number of jobs running concurrently during evaluation.

    Returns:
        (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
        K refers to the number of evaluation metrics.

    Raises:
        RuntimeError: If accumulation fails.
        ValueError: If ROI pruning is enabled but a dataset directory is not specified.
    """
    if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
        raise ValueError(
            "ROI pruning has been enabled, but the dataset directory has not be specified. "
            "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
        )

    # Sort both the detections and annotations by lexicographic order for grouping.
    dts = dts.sort_values(list(UUID_COLUMN_NAMES))
    gts = gts.sort_values(list(UUID_COLUMN_NAMES))

    dts_npy: NDArrayFloat = dts[list(DTS_COLUMN_NAMES)].to_numpy()
    gts_npy: NDArrayFloat = gts[list(GTS_COLUMN_NAMES)].to_numpy()

    dts_uuids: List[str] = dts[list(UUID_COLUMN_NAMES)].to_numpy().tolist()
    gts_uuids: List[str] = gts[list(UUID_COLUMN_NAMES)].to_numpy().tolist()

    # We merge the unique identifier -- the tuple of ("log_id", "timestamp_ns", "category")
    # into a single string to optimize the subsequent grouping operation.
    # `groupby_mapping` produces a mapping from the uuid to the group of detections / annotations
    # which fall into that group.
    uuid_to_dts = groupby([":".join(map(str, x)) for x in dts_uuids], dts_npy)
    uuid_to_gts = groupby([":".join(map(str, x)) for x in gts_uuids], gts_npy)

    log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

    # Load maps and egoposes if roi-pruning is enabled.
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        logger.info("Loading maps and egoposes ...")
        log_ids: List[str] = gts.loc[:, "log_id"].unique().tolist()
        log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(log_ids, cfg.dataset_dir)

    args_list: List[Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]] = []
    uuids = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
    for uuid in uuids:
        log_id, timestamp_ns, _ = uuid.split(":")
        args: Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]

        sweep_dts: NDArrayFloat = np.zeros((0, 10))
        sweep_gts: NDArrayFloat = np.zeros((0, 10))
        if uuid in uuid_to_dts:
            sweep_dts = uuid_to_dts[uuid]
        if uuid in uuid_to_gts:
            sweep_gts = uuid_to_gts[uuid]

        args = sweep_dts, sweep_gts, cfg, None, None
        if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
            avm = log_id_to_avm[log_id]
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            args = sweep_dts, sweep_gts, cfg, avm, city_SE3_ego
        args_list.append(args)

    logger.info("Starting evaluation ...")
    with get_context("spawn").Pool(processes=n_jobs) as p:
        outputs: Optional[List[Tuple[NDArrayFloat, NDArrayFloat]]] = p.starmap(accumulate, args_list)

    if outputs is None:
        raise RuntimeError("Accumulation has failed! Please check the integrity of your detections and annotations.")
    dts_list, gts_list = zip(*outputs)

    METRIC_COLUMN_NAMES = cfg.affinity_thresholds_m + TP_ERROR_COLUMNS + ("is_evaluated",)
    dts_metrics: NDArrayFloat = np.concatenate(dts_list)  # type: ignore
    gts_metrics: NDArrayFloat = np.concatenate(gts_list)  # type: ignore
    dts.loc[:, METRIC_COLUMN_NAMES] = dts_metrics
    gts.loc[:, METRIC_COLUMN_NAMES] = gts_metrics
    mean_average_recall, recall3d = summarize_metrics(dts, gts, cfg)
    print('mean_average_recall: ', mean_average_recall)
    return mean_average_recall

