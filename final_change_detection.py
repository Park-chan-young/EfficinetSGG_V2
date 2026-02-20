'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import uuid
from pathlib import Path
import pickle
import gzip
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import (Counter, defaultdict)
import open3d as o3d # New_script_Start

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox,
    orr_log_final_objs_pcd_and_obb, # New_script_Start 
    orr_log_final_objs_gaussians, # New_script_Start
    orr_log_final_objs_gaussian_splat_like, # New_script_Start
    orr_log_rgb_image, 
    orr_log_vlm_image,
    orr_log_final_objs_gaussian_saved_samples # New_script_Start
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import (
    get_dataset,
    load_before_objects
)
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections

####
def sample_pixels_from_mask_centerbiased(
    mask: np.ndarray,                 # (H,W) bool or {0,1}
    bbox_xyxy: Tuple[float,float,float,float],
    N: int,
    sigma_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    SAM mask 내부 픽셀만 대상으로 center-biased 샘플링.
    center는 bbox 중심을 쓰고, weight는 center와의 2D 거리 기반 Gaussian.
    반환: (M,2) int32, (x,y)
    """
    if mask is None:
        return np.zeros((0,2), dtype=np.int32)

    m = mask.astype(bool)
    ys, xs = np.where(m)
    if xs.size == 0:
        return np.zeros((0,2), dtype=np.int32)

    x1,y1,x2,y2 = map(float, bbox_xyxy)
    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
    w = max(1.0, x2-x1); h = max(1.0, y2-y1)
    sx = sigma_ratio*w; sy = sigma_ratio*h

    # center-biased weight
    dx = (xs - cx) / sx
    dy = (ys - cy) / sy
    wts = np.exp(-0.5*(dx*dx + dy*dy)).astype(np.float64)
    wts_sum = wts.sum()
    if not np.isfinite(wts_sum) or wts_sum <= 0:
        # fallback uniform
        idx = rng.choice(xs.size, size=min(N, xs.size), replace=(xs.size < N))
    else:
        p = wts / wts_sum
        idx = rng.choice(xs.size, size=min(N, xs.size), replace=(xs.size < N), p=p)

    xy = np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)
    return xy
####


####ew
# =========================
# Scene Change Detection (SAM mask 기반)
# =========================

@dataclass
class SceneCDCfg:
    # matching threshold (마스크 포인트 중 몇 %가 과거 object 가우시안에 "들어가면" 매칭으로 볼지)
    tau_inlier_match: float = 0.30     # 시작값. 너무 빡빡하면 0.2, 너무 느슨하면 0.4~0.5
    T_gate: float = 9.0               # Mahalanobis^2 gate (≈ 3-sigma)

    # label gating (선택)
    use_label_gating: bool = True
    tau_conf: float = 0.6             # conf >= tau_conf일 때만 label로 후보 제한
    # projection
    depth_scale: float = 1.0          # 너 코드가 depth 그대로 쓰는 흐름이면 1.0 (네가 말한 그대로)

    # 속도/메모리 안전장치 (원하면 None으로 두면 "전체" 사용)
    max_points_per_mask: Optional[int] = None  # 예: 20000 (너가 진짜 전체 원하면 None)

    eps: float = 1e-6


def _build_gaussian_cache(before_objects: List[Dict[str, Any]], eps: float = 1e-6):
    """
    before_objects[object_index]["gaussians"] 안의 (mu,Sigma)를 읽어서
    inv(Sigma) 캐시 + class_name -> object_index 리스트 맵을 만든다.
    """
    gauss_cache: List[List[Dict[str, np.ndarray]]] = []
    label_to_obj: Dict[str, List[int]] = {}

    for obj_idx, obj in enumerate(before_objects):
        cls = str(obj.get("class_name", "unknown"))
        label_to_obj.setdefault(cls, []).append(obj_idx)

        per_obj = []
        for g in obj.get("gaussians", []):
            mu = np.asarray(g["mu"], dtype=np.float32).reshape(3)
            Sigma = np.asarray(g["Sigma"], dtype=np.float32).reshape(3, 3)
            Sigma = 0.5 * (Sigma + Sigma.T) + np.eye(3, dtype=np.float32) * eps
            invS = np.linalg.inv(Sigma).astype(np.float32)
            per_obj.append({"mu": mu, "invS": invS})
        gauss_cache.append(per_obj)

    return gauss_cache, label_to_obj


def _mask_all_pixels(mask_hw: np.ndarray) -> np.ndarray:
    """
    mask_hw: (H,W) bool/0-1
    return: (M,2) with columns [x,y]
    """
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _project_pixels_to_world(
    pixels_xy: np.ndarray,          # (M,2) int [x,y]
    depth_array: np.ndarray,         # (H,W)
    intrinsics_4x4: np.ndarray,      # (4,4) (너 코드 intrinsics)
    pose_4x4: np.ndarray,            # (4,4) T_world_cam (네 코드 adjusted_pose 흐름과 동일)
    depth_scale: float
) -> np.ndarray:
    """
    mask 내부 전체 픽셀을 depth로 3D화하고 world로 변환.
    return: (P,3) world points (depth invalid 제거됨)
    """
    H, W = depth_array.shape[:2]
    if pixels_xy.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xs = pixels_xy[:, 0]
    ys = pixels_xy[:, 1]
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    d = depth_array[ys, xs].astype(np.float32) * float(depth_scale)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    d = d[valid]

    fx = float(intrinsics_4x4[0, 0])
    fy = float(intrinsics_4x4[1, 1])
    cx = float(intrinsics_4x4[0, 2])
    cy = float(intrinsics_4x4[1, 2])

    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    Z = d

    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # (P,4)
    pts_w = (pose_4x4 @ pts_cam.T).T[:, :3].astype(np.float32)  # (P,3)
    return pts_w


def _inlier_ratio_to_object(
    pts_w: np.ndarray,                  # (P,3)
    obj_gauss_cache: List[Dict[str, np.ndarray]],
    T_gate: float
) -> float:
    """
    각 point에 대해 min_k Mahalanobis^2 계산해서 gate 통과하면 inlier.
    """
    if pts_w.shape[0] == 0 or len(obj_gauss_cache) == 0:
        return 0.0

    inlier = 0
    # brute-force (P * K). K는 작고(예: 5~30) P는 mask 포인트라 클 수 있음
    for p in pts_w:
        best_d2 = np.inf
        for g in obj_gauss_cache:
            diff = p - g["mu"]
            d2 = float(diff.T @ g["invS"] @ diff)
            if d2 < best_d2:
                best_d2 = d2
        if best_d2 < T_gate:
            inlier += 1

    return float(inlier) / float(pts_w.shape[0])


def scene_cd_per_frame_from_gobs(
    frame_idx: int,
    gobs: Dict[str, Any],
    depth_array: np.ndarray,
    intrinsics_4x4: np.ndarray,
    pose_4x4: np.ndarray,
    before_objects: List[Dict[str, Any]],
    gauss_cache: List[List[Dict[str, np.ndarray]]],
    label_to_obj: Dict[str, List[int]],
    cfg: SceneCDCfg,
    rng: np.random.Generator,
    alive_flags: np.ndarray,  # (num_before_objects,) 0/1
) -> Tuple[List[str], Dict[str, Any]]:
    """
    반환:
      - unmatched_class_labels: List[str]  (이 프레임에서 과거 object로 매칭 안 된 mask들의 class label들)
      - frame_stat: dict (디버깅용)
    """
    unmatched_labels: List[str] = []
    stat = {"frame_idx": frame_idx, "num_masks": 0, "num_matched": 0, "num_unmatched": 0}

    masks = gobs.get("mask", None)
    xyxy = gobs.get("xyxy", None)
    confs = gobs.get("confidence", None)
    class_ids = gobs.get("class_id", None)
    classes_arr = gobs.get("classes", None)  # class id -> name 배열

    if masks is None or len(masks) == 0:
        return unmatched_labels, stat

    stat["num_masks"] = int(len(masks))

    # classes_arr는 너 결과 dict에 넣어둔 값 그대로 쓰면 됨.
    # results에 "classes": obj_classes.get_classes_arr() 들어가 있음 (네가 준 results 구조).
    def _cls_name_from_det(i: int) -> str:
        if classes_arr is None or class_ids is None:
            return "unknown"
        cid = int(class_ids[i])
        if 0 <= cid < len(classes_arr):
            return str(classes_arr[cid])
        return str(cid)

    for i in range(len(masks)):
        mask_hw = masks[i]
        if mask_hw is None:
            continue

        # 2) mask 내부 전체 픽셀 좌표
        pix = _mask_all_pixels(mask_hw)

        # (선택) 너무 많으면 cap (원하면 cfg.max_points_per_mask=None로 전체)
        if cfg.max_points_per_mask is not None and pix.shape[0] > cfg.max_points_per_mask:
            sel = rng.choice(pix.shape[0], size=int(cfg.max_points_per_mask), replace=False)
            pix = pix[sel]

        # 3) 2D->3D->world
        pts_w = _project_pixels_to_world(
            pixels_xy=pix,
            depth_array=depth_array,
            intrinsics_4x4=intrinsics_4x4,
            pose_4x4=pose_4x4,
            depth_scale=cfg.depth_scale
        )

        if pts_w.shape[0] == 0:
            unmatched_labels.append(_cls_name_from_det(i))
            continue

        # 3)~4) 과거 object들과 매칭: 가장 높은 inlier_ratio를 주는 object 선택
        det_conf = float(confs[i]) if confs is not None else 0.0
        det_label = _cls_name_from_det(i)

        # label gating(옵션)
        candidate_obj_indices = list(range(len(before_objects)))
        if cfg.use_label_gating and (det_conf >= cfg.tau_conf):
            cand = label_to_obj.get(det_label, [])
            if len(cand) > 0:
                candidate_obj_indices = cand  # 같은 class object들만
            # 없으면 fallback으로 전체

        best_obj = -1
        best_ratio = -1.0
        for obj_idx in candidate_obj_indices:
            r = _inlier_ratio_to_object(pts_w, gauss_cache[obj_idx], cfg.T_gate)
            if r > best_ratio:
                best_ratio = r
                best_obj = obj_idx

        # 매칭 판정
        if best_ratio >= cfg.tau_inlier_match and best_obj >= 0:
            alive_flags[best_obj] = 1  # 4) 살아있는 object index 체크
            stat["num_matched"] += 1
        else:
            unmatched_labels.append(det_label)
            stat["num_unmatched"] += 1

    # 중복 제거(프레임 출력용)
    if len(unmatched_labels) > 1:
        unmatched_labels = sorted(list(set(unmatched_labels)))

    return unmatched_labels, stat


# =========================
# Scene Change Detection v3 (2D mask center 기반, object-centric)
# =========================

@dataclass
class MaskCenterCDCfg:
    """2D mask 중심점 기반 change detection 설정.

    (A) Created(새로 생긴 것):
      - 2D mask에서 중심점(pixel)을 구한 뒤, 그 pixel의 depth로 3D point를 만든다.
      - 그 3D point 주변에 같은 class label의 기존 Gaussian(mu)이 없으면 created.

    (B) Disappeared(사라진 것):
      - 프레임마다 각 detection mask의 2D 중심점 -> 3D로 변환
      - 같은 class label이고, 그 3D point가 기존 Gaussian(mu)들 중 하나와 일정 거리 이내면
        해당 gaussian이 속한 object_index를 "관측됨"으로 표시한다.
      - 시퀀스 전체를 처리한 뒤, 한 번도 관측되지 않은 before object_index를 disappeared로 출력한다.
    """

    # ---------- created ----------
    tau_new_gaussian_dist: float = 0.25      # [m] center point 주변에 같은 class gaussian(mu)이 이 거리 이내 없으면 created
    use_det_conf_gate_for_created: bool = True
    tau_conf_created: float = 0.6

    # ---------- matching (for disappeared tracking) ----------
    tau_match_gaussian_dist: float = 0.25    # [m] center point가 기존 gaussian(mu)와 이 거리 이내면 match
    use_det_conf_gate_for_match: bool = False
    tau_conf_match: float = 0.0

    # ---------- projection ----------
    depth_scale: float = 1.0
    center_depth_ksize: int = 5             # 중심 pixel depth가 invalid일 때 주변 window median depth 사용
    min_depth: float = 1e-6                 # depth invalid threshold
    eps: float = 1e-6


@dataclass
class BeforeSceneGaussianCache:
    """before_objects에서 뽑아낸 cache (1회 빌드 후 프레임마다 재사용)."""

    obj_class: List[str]
    obj_ids: List[int]  # internal index -> gaussians_meta의 object_index (없으면 내부 idx)
    class_to_obj_indices: Dict[str, List[int]]

    # class별 gaussian center(mu) KDTree + 그 gaussian이 속한 object 내부 index mapping
    class_to_gauss_xyz: Dict[str, np.ndarray]         # class -> (Ng,3)
    class_to_gauss_objidx: Dict[str, np.ndarray]      # class -> (Ng,) int (internal object index)
    class_to_gauss_tree: Dict[str, cKDTree]           # class -> KDTree

def build_before_scene_gaussian_cache(
    before_objects: List[Dict[str, Any]],
    eps: float = 1e-6,
) -> BeforeSceneGaussianCache:
    """before_objects에서 class별 gaussian center(mu) KDTree + object_index mapping을 만든다.

    - internal object index: before_objects list index (0..N-1)
    - obj_ids: gaussians_meta의 object_index를 보존(가능하면). 없으면 internal index 사용
    """

    obj_class: List[str] = []
    obj_ids: List[int] = []
    class_to_obj_indices: Dict[str, List[int]] = defaultdict(list)

    class_to_gauss_list: Dict[str, List[np.ndarray]] = defaultdict(list)
    class_to_gauss_obj_list: Dict[str, List[int]] = defaultdict(list)

    for obj_idx, obj in enumerate(before_objects):
        cls = str(obj.get("class_name", "unknown"))

        # object_id(=gaussians_meta의 object_index)를 우선 사용
        obj_id = obj.get("object_index", None)
        if obj_id is None:
            obj_id = (obj.get("object_meta", {}) or {}).get("object_index", None)
        if obj_id is None:
            obj_id = int(obj_idx)

        obj_class.append(cls)
        obj_ids.append(int(obj_id))
        class_to_obj_indices[cls].append(int(obj_idx))

        for g in obj.get("gaussians", []) or []:
            if "mu" not in g:
                continue
            mu = np.asarray(g["mu"], dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(mu)):
                continue
            class_to_gauss_list[cls].append(mu)
            # KDTree 매칭은 internal index로 alive_flags를 업데이트하기 위해 internal idx를 저장
            class_to_gauss_obj_list[cls].append(int(obj_idx))

    class_to_gauss_xyz: Dict[str, np.ndarray] = {}
    class_to_gauss_objidx: Dict[str, np.ndarray] = {}
    class_to_gauss_tree: Dict[str, cKDTree] = {}

    for cls, mus in class_to_gauss_list.items():
        if len(mus) == 0:
            continue
        arr = np.stack(mus, axis=0).astype(np.float32)
        class_to_gauss_xyz[cls] = arr
        class_to_gauss_objidx[cls] = np.asarray(class_to_gauss_obj_list[cls], dtype=np.int32)
        class_to_gauss_tree[cls] = cKDTree(arr)

    return BeforeSceneGaussianCache(
        obj_class=obj_class,
        obj_ids=obj_ids,
        class_to_obj_indices=dict(class_to_obj_indices),
        class_to_gauss_xyz=class_to_gauss_xyz,
        class_to_gauss_objidx=class_to_gauss_objidx,
        class_to_gauss_tree=class_to_gauss_tree,
    )

def _mask_center_pixel(mask_hw: np.ndarray) -> Optional[Tuple[int, int]]:
    """mask에서 2D 중심점(pixel)을 계산한다. return (cx, cy) int."""
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0:
        return None
    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))
    return cx, cy


def _depth_at_pixel_with_fallback(
    depth_array: np.ndarray,
    cx: int,
    cy: int,
    mask_hw: Optional[np.ndarray],
    ksize: int,
    min_depth: float,
) -> Optional[float]:
    """(cx,cy)에서 depth가 invalid면 주변 window median으로 대체."""
    H, W = depth_array.shape[:2]
    cx = int(np.clip(cx, 0, W - 1))
    cy = int(np.clip(cy, 0, H - 1))

    d0 = float(depth_array[cy, cx])
    if np.isfinite(d0) and d0 > float(min_depth):
        return d0

    # fallback: 주변 window에서 median depth
    r = int(max(0, ksize // 2))
    x0 = max(0, cx - r)
    x1 = min(W, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(H, cy + r + 1)

    sub = depth_array[y0:y1, x0:x1]
    if sub.size == 0:
        return None

    if mask_hw is not None:
        sub_mask = (mask_hw[y0:y1, x0:x1] > 0)
        vals = sub[sub_mask]
    else:
        vals = sub.reshape(-1)

    vals = vals[np.isfinite(vals) & (vals > float(min_depth))]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def _project_center_pixel_to_world(
    cx: int,
    cy: int,
    mask_hw: np.ndarray,
    depth_array: np.ndarray,
    intrinsics_4x4: np.ndarray,
    pose_4x4: np.ndarray,
    depth_scale: float,
    center_depth_ksize: int,
    min_depth: float,
) -> Optional[np.ndarray]:
    """2D 중심 pixel (cx,cy)을 depth+intrinsics+pose로 world 3D point로 변환."""
    d_raw = _depth_at_pixel_with_fallback(
        depth_array=depth_array,
        cx=cx,
        cy=cy,
        mask_hw=mask_hw,
        ksize=center_depth_ksize,
        min_depth=min_depth,
    )
    if d_raw is None:
        return None

    d = float(d_raw) / float(depth_scale)

    fx = float(intrinsics_4x4[0, 0])
    fy = float(intrinsics_4x4[1, 1])
    cx0 = float(intrinsics_4x4[0, 2])
    cy0 = float(intrinsics_4x4[1, 2])

    if fx <= 0 or fy <= 0:
        return None

    X = (float(cx) - cx0) * d / fx
    Y = (float(cy) - cy0) * d / fy
    Z = d

    pt_c = np.asarray([X, Y, Z, 1.0], dtype=np.float32)
    pt_w = (pose_4x4 @ pt_c)[:3].astype(np.float32)
    if not np.all(np.isfinite(pt_w)):
        return None
    return pt_w


def mask_center_cd_per_frame(
    frame_idx: int,
    gobs: Dict[str, Any],
    depth_array: np.ndarray,
    intrinsics_4x4: np.ndarray,
    pose_4x4: np.ndarray,
    before_cache: BeforeSceneGaussianCache,
    cfg: MaskCenterCDCfg,
)-> Tuple[List[str], List[int], Dict[str, Any]]:
    """프레임 단위 change detection.

    반환:
      - created_labels: 이 프레임에서 created로 판정된 class label (중복 제거)
      - matched_object_indices: 이 프레임에서 기존 object로 매칭된 before object_index 목록(중복 제거)
      - stat: 디버깅 정보
    """

    masks = gobs.get("mask", None)
    confs = gobs.get("confidence", None)
    class_ids = gobs.get("class_id", None)
    classes_arr = gobs.get("classes", None)

    stat: Dict[str, Any] = {
        "frame_idx": int(frame_idx),
        "num_masks": 0,
        "created": [],
        "matched_objects": [],  # list of {object_index, class_name, dist}
    }

    if masks is None or len(masks) == 0:
        return [], [], stat

    stat["num_masks"] = int(len(masks))

    def _cls_name_from_det(i: int) -> str:
        if classes_arr is None or class_ids is None:
            return "unknown"
        cid = int(class_ids[i])
        if 0 <= cid < len(classes_arr):
            return str(classes_arr[cid])
        return str(cid)

    created_labels: List[str] = []
    matched_obj_indices: List[int] = []

    for i in range(len(masks)):
        mask_hw = masks[i]
        if mask_hw is None:
            continue

        cls = _cls_name_from_det(i)
        conf = float(confs[i]) if confs is not None else 0.0

        center = _mask_center_pixel(mask_hw)
        if center is None:
            continue
        cx, cy = center

        center_w = _project_center_pixel_to_world(
            cx=cx,
            cy=cy,
            mask_hw=mask_hw,
            depth_array=depth_array,
            intrinsics_4x4=intrinsics_4x4,
            pose_4x4=pose_4x4,
            depth_scale=cfg.depth_scale,
            center_depth_ksize=cfg.center_depth_ksize,
            min_depth=cfg.min_depth,
        )
        if center_w is None:
            continue

        tree = before_cache.class_to_gauss_tree.get(cls, None)
        if tree is None:
            # before scene에 해당 class 자체가 없으면 created
            if (not cfg.use_det_conf_gate_for_created) or (conf >= float(cfg.tau_conf_created)):
                created_labels.append(cls)
            continue

        # nearest gaussian in same class
        d, nn_idx = tree.query(center_w.astype(np.float32), k=1)
        d = float(d)
        nn_idx = int(nn_idx)

        # (B) matching -> alive tracking (disappeared용)
        if (not cfg.use_det_conf_gate_for_match) or (conf >= float(cfg.tau_conf_match)):
            if d <= float(cfg.tau_match_gaussian_dist):
                obj_idx = int(before_cache.class_to_gauss_objidx[cls][nn_idx])
                matched_obj_indices.append(obj_idx)
                stat["matched_objects"].append({
                    "object_index": int(obj_idx),
                    "class_name": str(cls),
                    "dist": float(d),
                })

        # (A) created 판정
        if (not cfg.use_det_conf_gate_for_created) or (conf >= float(cfg.tau_conf_created)):
            if d > float(cfg.tau_new_gaussian_dist):
                created_labels.append(cls)

    # unique (preserve order)
    if len(created_labels) > 0:
        seen = set(); tmp = []
        for c in created_labels:
            if c not in seen:
                tmp.append(c); seen.add(c)
        created_labels = tmp
    stat["created"] = created_labels

    if len(matched_obj_indices) > 0:
        seen = set(); tmp = []
        for oi in matched_obj_indices:
            if oi not in seen:
                tmp.append(int(oi)); seen.add(int(oi))
        matched_obj_indices = tmp

    return created_labels, matched_obj_indices, stat


# =========================
# Scene Change Detection v4 (2D mask center + CLIP feature similarity, object-centric)
# =========================

@dataclass
class ClipCenterCDCfg:
    """CLIP feature 기반 1차 매칭 + 2D mask 중심점(픽셀) -> 3D point 기반 CD.

    - 기존(v3): class label로 후보 gaussian을 제한
    - 변경(v4): detection의 CLIP image feature와 before object의 CLIP feature cosine similarity로 후보를 만들고,
              그 후보 object의 gaussian centers(mu)와의 거리로 match/created를 판단.

    반환은 created 검출 인덱스까지 포함해서, 이후 created mask를 3D로 투영해 시퀀스 전체 3D object로 병합할 수 있게 한다.
    """

    # --- CLIP gating ---
    clip_topk: int = 5               # similarity 상위 K개 object만 거리 검사
    tau_clip_sim: float = 0.25       # 후보로 인정할 최소 cosine sim

    # --- background skip (e.g., wall/ceiling) ---
    # Change detection에서 배경 클래스는 created/matched 판단에서 제외한다.
    skip_bg_in_cd: bool = True
    bg_class_names: Tuple[str, ...] = ("wall", "ceiling", "floor", "window", "door") # 벽, 천장, 바닥, 창문, 문 같은 배경은 제외 객체 중심으로

    # --- created ---
    tau_new_gaussian_dist: float = 0.25
    use_det_conf_gate_for_created: bool = True
    tau_conf_created: float = 0.6

    # --- matching for disappeared tracking ---
    tau_match_gaussian_dist: float = 0.25
    use_det_conf_gate_for_match: bool = False
    tau_conf_match: float = 0.0

    # --- score fusion for alive matching (disappeared) ---
    # 후보 object들에 대해 sim + 거리(d)를 하나의 score로 결합해 alive를 결정한다.
    # score_j = w_d * (d / tau_match_gaussian_dist) + w_s * (1 - sim_j)
    # (선택) use_sim_norm_in_score=True이면 sim을 [0,1]로 정규화해서 penalty로 사용한다.
    use_score_fusion_for_match: bool = True
    use_sim_norm_in_score: bool = False
    w_dist_match: float = 1.0
    w_sim_match: float = 2.0
    tau_score_match: float = 0.8          # score가 이 값 이하면 alive로 인정
    max_alive_per_det: int = 0            # 0이면 제한 없음, >0이면 score가 좋은 상위 M개만 alive로 처리

    # --- projection ---
    depth_scale: float = 1.0
    center_depth_ksize: int = 5
    min_depth: float = 1e-6
    eps: float = 1e-6


@dataclass
class BeforeSceneClipGaussianCache:
    """before_objects에서 CLIP feature + object별 gaussian center KDTree를 만든 cache."""

    obj_class: List[str]
    obj_ids: List[int]                     # internal object idx -> external object_index (gaussians_meta)
    obj_clip_feats: np.ndarray             # (N,D) float32, L2-normalized
    obj_clip_valid: np.ndarray             # (N,) bool
    obj_to_gauss_obj_tree: List[Optional[cKDTree]]  # per object KDTree over gaussian centers
    obj_has_gaussians: np.ndarray          # (N,) bool



@dataclass
class Created3DConfirmCfg:
    """시퀀스 전체 created 3D 객체 확정(confirmation) 설정.

    created로 1~2번만 튀는 노이즈 detection을 최종 3D 객체로 남기지 않기 위해 사용한다.

    - min_consecutive_frames:
        어떤 created object가 연속된 프레임에서 최소 몇 번 이상 관측되어야 유지할지.
        (예: 3이면 ... t,t+1,t+2에서라도 한 번 연속으로 잡혀야 유지)
    - min_total_detections:
        전체 시퀀스 누적 검출 횟수 최소값(선택적 안전장치).
    """
    min_consecutive_frames: int = 5
    min_total_detections: int = 10

def _to_numpy_feat(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    if x.size == 0:
        return None
    if x.ndim > 1:
        x = x.reshape(-1)
    return x.astype(np.float32, copy=False)


def _l2_normalize(v: np.ndarray, eps: float = 1e-6) -> Optional[np.ndarray]:
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < float(eps):
        return None
    return (v / n).astype(np.float32, copy=False)


def build_before_scene_clip_gaussian_cache(
    before_objects: List[Dict[str, Any]],
    eps: float = 1e-6,
) -> BeforeSceneClipGaussianCache:
    """before_objects에서 object-level CLIP feature와 object별 gaussian center KDTree를 만든다."""

    obj_class: List[str] = []
    obj_ids: List[int] = []
    clip_list: List[Optional[np.ndarray]] = []
    gauss_trees: List[Optional[cKDTree]] = []
    has_gauss: List[bool] = []

    # 먼저 feature dim 추정
    D = None
    for obj in before_objects:
        ft = obj.get("clip_ft", None)
        if ft is None:
            ft = obj.get("clip_feature", None)
        if ft is None:
            ft = obj.get("clip_features", None)
        if ft is None:
            ft = obj.get("clip", None)
        ft = _to_numpy_feat(ft)
        ft = _l2_normalize(ft, eps=eps)
        if ft is not None:
            D = int(ft.shape[0])
            break
    if D is None:
        raise ValueError("before_objects에서 object-level clip feature를 찾지 못했습니다. (예: obj['clip_ft'])")

    for obj_idx, obj in enumerate(before_objects):
        cls = str(obj.get("class_name", "unknown"))

        obj_id = obj.get("object_index", None)
        if obj_id is None:
            obj_id = (obj.get("object_meta", {}) or {}).get("object_index", None)
        if obj_id is None:
            obj_id = int(obj_idx)

        obj_class.append(cls)
        obj_ids.append(int(obj_id))

        # object-level clip feature
        ft = obj.get("clip_ft", None)
        if ft is None:
            ft = obj.get("clip_feature", None)
        if ft is None:
            ft = obj.get("clip_features", None)
        if ft is None:
            ft = obj.get("clip", None)

        ft = _to_numpy_feat(ft)
        ft = _l2_normalize(ft, eps=eps)
        if ft is None:
            clip_list.append(None)
        else:
            clip_list.append(ft)

        # object-level gaussian centers KDTree
        mus = []
        for g in obj.get("gaussians", []) or []:
            if "mu" not in g:
                continue
            mu = np.asarray(g["mu"], dtype=np.float32).reshape(3)
            if np.all(np.isfinite(mu)):
                mus.append(mu)

        if len(mus) > 0:
            arr = np.stack(mus, axis=0).astype(np.float32)
            gauss_trees.append(cKDTree(arr))
            has_gauss.append(True)
        else:
            gauss_trees.append(None)
            has_gauss.append(False)

    # stack clip feats
    obj_clip_feats = np.zeros((len(before_objects), D), dtype=np.float32)
    obj_clip_valid = np.zeros((len(before_objects),), dtype=bool)
    for i, ft in enumerate(clip_list):
        if ft is None:
            continue
        if ft.shape[0] != D:
            continue
        obj_clip_feats[i] = ft
        obj_clip_valid[i] = True

    return BeforeSceneClipGaussianCache(
        obj_class=obj_class,
        obj_ids=obj_ids,
        obj_clip_feats=obj_clip_feats,
        obj_clip_valid=obj_clip_valid,
        obj_to_gauss_obj_tree=gauss_trees,
        obj_has_gaussians=np.asarray(has_gauss, dtype=bool),
    )


def _slice_gobs_by_indices(gobs: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
    """gobs dict를 detection indices로 슬라이싱한다."""
    if gobs is None:
        return {}
    if indices is None or len(indices) == 0:
        return {}

    idx = np.asarray(indices, dtype=np.int32)
    N = None
    if "mask" in gobs and gobs["mask"] is not None:
        try:
            N = len(gobs["mask"])
        except Exception:
            N = None

    out: Dict[str, Any] = {}
    for k, v in gobs.items():
        if k == "classes":
            out[k] = v
            continue
        if v is None:
            out[k] = None
            continue

        # numpy arrays
        if isinstance(v, np.ndarray):
            if N is not None and v.shape[0] == N:
                out[k] = v[idx]
            else:
                out[k] = v
            continue

        # lists/tuples aligned with detections
        if isinstance(v, (list, tuple)) and (N is not None) and (len(v) == N):
            out[k] = [v[int(i)] for i in idx.tolist()]
            continue

        out[k] = v

    return out


def clip_center_cd_per_frame(
    frame_idx: int,
    gobs: Dict[str, Any],
    depth_array: np.ndarray,
    intrinsics_4x4: np.ndarray,
    pose_4x4: np.ndarray,
    before_cache: BeforeSceneClipGaussianCache,
    cfg: ClipCenterCDCfg,
) -> Tuple[List[str], List[int], List[int], Dict[str, Any]]:
    """프레임 단위 CD (CLIP 기반 1차 매칭).

    반환:
      - created_labels: 이 프레임에서 created로 판정된 label들(중복 제거)
      - created_det_indices: created로 판정된 detection index들
      - matched_object_indices: 기존 before object로 매칭된 internal object index들 (alive update용)
      - stat: 디버깅 정보
    """

    masks = gobs.get("mask", None)
    confs = gobs.get("confidence", None)
    class_ids = gobs.get("class_id", None)
    classes_arr = gobs.get("classes", None)
    image_feats = gobs.get("image_feats", None)

    stat: Dict[str, Any] = {
        "frame_idx": int(frame_idx),
        "num_masks": 0,
        "created": [],
        "created_det_indices": [],
        "matched_objects": [],   # list of {internal_obj_idx, object_index, dist, sim}
    }

    if masks is None or len(masks) == 0:
        return [], [], [], stat
    stat["num_masks"] = int(len(masks))

    def _cls_name_from_det(i: int) -> str:
        if classes_arr is None or class_ids is None:
            return "unknown"
        cid = int(class_ids[i])
        if 0 <= cid < len(classes_arr):
            return str(classes_arr[cid])
        return str(cid)

    created_labels: List[str] = []
    created_det_indices: List[int] = []
    matched_obj_indices: List[int] = []

    # pre-fetch before feats
    B = before_cache.obj_clip_feats
    valid_mask = before_cache.obj_clip_valid
    Nobj = B.shape[0]

    # background classes to skip in change detection (e.g., wall/ceiling/floor)
    bg_set = set()
    if getattr(cfg, "skip_bg_in_cd", False):
        try:
            bg_set = {str(x).strip().lower() for x in (cfg.bg_class_names or ())}
        except Exception:
            bg_set = set()


    for i in range(len(masks)):
        mask_hw = masks[i]
        if mask_hw is None:
            continue

        conf = float(confs[i]) if confs is not None else 0.0
        cls_hint = _cls_name_from_det(i)

        # Skip background classes for CD (requested: wall/ceiling/floor 등 제외)
        if bg_set and str(cls_hint).strip().lower() in bg_set:
            continue

        center = _mask_center_pixel(mask_hw)
        if center is None:
            continue
        cx, cy = center

        center_w = _project_center_pixel_to_world(
            cx=cx,
            cy=cy,
            mask_hw=mask_hw,
            depth_array=depth_array,
            intrinsics_4x4=intrinsics_4x4,
            pose_4x4=pose_4x4,
            depth_scale=cfg.depth_scale,
            center_depth_ksize=cfg.center_depth_ksize,
            min_depth=cfg.min_depth,
        )
        if center_w is None:
            continue

        # detection clip feature
        det_ft = None
        try:
            if image_feats is not None:
                det_ft = image_feats[i]
        except Exception:
            det_ft = None

        det_ft = _to_numpy_feat(det_ft)
        det_ft = _l2_normalize(det_ft, eps=cfg.eps)
        if det_ft is None or Nobj == 0:
            # feature를 못 얻으면 판단 불가 -> skip
            continue

        # cosine similarity to all before objects (normalized => dot product)
        sims = (B @ det_ft.reshape(-1, 1)).reshape(-1)  # (Nobj,)
        # invalidate missing feats
        sims = np.where(valid_mask, sims, -np.inf)

        # top-k candidates by similarity
        k = int(max(1, min(int(cfg.clip_topk), Nobj)))
        if k < Nobj:
            cand = np.argpartition(-sims, k - 1)[:k]
        else:
            cand = np.arange(Nobj, dtype=np.int32)

        # keep only sims >= tau
        cand = [int(j) for j in cand.tolist() if float(sims[int(j)]) >= float(cfg.tau_clip_sim)]
        best_obj = -1
        best_sim = -np.inf
        best_d = float("inf")

        # (B) matching -> alive tracking (disappeared용)
        # NOTE: 기존에는 후보 중 "가장 가까운 1개(best_obj)"만 alive 처리했지만,
        #       disappeared 허수를 줄이기 위해 1-to-many로 변경:
        #       CLIP(sim>=tau_clip_sim) 후보들 중 거리까지 통과한 모든 object를 alive로 표시한다.
        allow_match = (not cfg.use_det_conf_gate_for_match) or (conf >= float(cfg.tau_conf_match))

        # score-based 1-to-many alive update (sim + 거리 통합)
        tau_match = float(cfg.tau_match_gaussian_dist)
        if not np.isfinite(tau_match) or tau_match <= 0.0:
            tau_match = 1.0
        tau_clip = float(cfg.tau_clip_sim)
        denom_sim = (1.0 - tau_clip) if (1.0 - tau_clip) > 1e-6 else 1.0

        passed = []  # list of (score, internal_obj_idx, dist, sim)

        for j in cand:
            tree = before_cache.obj_to_gauss_obj_tree[j]
            if tree is None:
                continue
            d, _ = tree.query(center_w.astype(np.float32), k=1)
            d = float(d)
            sim_j = float(sims[int(j)])

            # best match (created 판정에 사용: 거리 기준)
            if d < best_d:
                best_d = d
                best_obj = int(j)
                best_sim = sim_j

            if not allow_match:
                continue

            # hard distance gate (물리적으로 너무 멀면 제외)
            if d > tau_match:
                continue

            if getattr(cfg, "use_score_fusion_for_match", True):
                # sim을 [0,1] 범위로 정규화해서 penalty로 사용 (tau_clip_sim도 반영)
                if getattr(cfg, "use_sim_norm_in_score", False):
                    sim_norm = (sim_j - tau_clip) / denom_sim
                    if sim_norm < 0.0:
                        sim_norm = 0.0
                    elif sim_norm > 1.0:
                        sim_norm = 1.0
                    sim_cost = 1.0 - sim_norm
                else:
                    sim_cost = 1.0 - sim_j

                d_norm = d / tau_match
                score = float(getattr(cfg, "w_dist_match", 1.0)) * d_norm + float(getattr(cfg, "w_sim_match", 1.0)) * sim_cost

                if score <= float(getattr(cfg, "tau_score_match", 1.0)):
                    passed.append((float(score), int(j), float(d), float(sim_j)))
            else:
                # legacy: 거리만으로 alive 처리
                passed.append((0.0, int(j), float(d), float(sim_j)))

        # keep only top-M by score if requested
        max_m = int(getattr(cfg, "max_alive_per_det", 0))
        if max_m > 0 and len(passed) > max_m:
            passed.sort(key=lambda x: x[0])
            passed = passed[:max_m]

        for score, jj, d, sim_j in passed:
            matched_obj_indices.append(int(jj))
            stat["matched_objects"].append({
                "internal_obj_idx": int(jj),
                "object_index": int(before_cache.obj_ids[int(jj)]),
                "dist": float(d),
                "sim": float(sim_j),
                "score": float(score),
            })
# (A) created 판정
        if (not cfg.use_det_conf_gate_for_created) or (conf >= float(cfg.tau_conf_created)):
            # 후보가 없거나, 후보 중 가장 가까운 gaussian까지의 거리가 크면 created
            if (best_obj < 0) or (best_d > float(cfg.tau_new_gaussian_dist)):
                created_labels.append(str(cls_hint))
                created_det_indices.append(int(i))

    # unique created labels (preserve order)
    if len(created_labels) > 0:
        seen = set(); tmp = []
        for c in created_labels:
            if c not in seen:
                tmp.append(c); seen.add(c)
        created_labels = tmp

    # unique created det indices (preserve order)
    if len(created_det_indices) > 0:
        seen = set(); tmp = []
        for di in created_det_indices:
            if int(di) not in seen:
                tmp.append(int(di)); seen.add(int(di))
        created_det_indices = tmp

    # unique matched obj indices
    if len(matched_obj_indices) > 0:
        seen = set(); tmp = []
        for oi in matched_obj_indices:
            if int(oi) not in seen:
                tmp.append(int(oi)); seen.add(int(oi))
        matched_obj_indices = tmp

    stat["created"] = created_labels
    stat["created_det_indices"] = created_det_indices
    return created_labels, created_det_indices, matched_obj_indices, stat


####ew
# New_script_Start_Coasrse 

# =========================
# Config
# =========================
@dataclass
class CoarseCDCfg:
    # sampling
    N: int = 50
    sigma: float = 0.22
    min_valid: int = 5
    max_resample_tries: int = 500

    # depth filter
    use_local_median: bool = True
    median_ksize: int = 3
    delta_depth: float = 0.05   # depth 값 스케일에 맞게(네 코드와 동일 단위 기준)

    # gating
    T_gate: float = 9.0
    tau_inlier: float = 0.3

    # label optional gating
    tau_conf: float = 0.6  # class name mismatch 심하면 1.1로 올려서 사실상 label-gating OFF

    # scale/pose
    depth_scale: float = 1.0        # ✅ 너 코드가 depth 그대로 쓰므로 1.0
    pose_is_T_world_cam: bool = True # ✅ pcd.transform(adjusted_pose) 그대로 쓰는 흐름과 일치

    eps: float = 1e-6


# =========================
# Cache builder
# =========================
def build_gaussian_cache(before_objects: List[Dict[str, Any]], eps: float = 1e-6):
    """
    before_objects 기반으로 inv(Sigma) 캐시 생성 + class_name->object_indices 맵 생성
    """
    gauss_cache: List[List[Dict[str, np.ndarray]]] = []
    label_to_obj: Dict[str, List[int]] = {}

    for obj_idx, obj in enumerate(before_objects):
        cls = str(obj.get("class_name", "unknown"))
        label_to_obj.setdefault(cls, []).append(obj_idx)

        per_obj = []
        for g in obj.get("gaussians", []):
            mu = np.asarray(g["mu"], dtype=np.float32).reshape(3)
            Sigma = np.asarray(g["Sigma"], dtype=np.float32).reshape(3, 3)
            Sigma = 0.5 * (Sigma + Sigma.T) + np.eye(3, dtype=np.float32) * eps
            invS = np.linalg.inv(Sigma).astype(np.float32)
            per_obj.append({"mu": mu, "invS": invS})
        gauss_cache.append(per_obj)

    return gauss_cache, label_to_obj


# =========================
# Coarse-CD core utils
# =========================
def center_biased_sample_pixels(
    bbox_xyxy: Tuple[float, float, float, float],
    H: int, W: int,
    N: int,
    sigma_ratio: float,
    rng: np.random.Generator,
    max_tries: int
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    x1 = max(0.0, min(float(W - 1), x1)); x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1)); y2 = max(0.0, min(float(H - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 2), dtype=np.int32)

    cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    sx = sigma_ratio * w;  sy = sigma_ratio * h

    pts = []
    tries = 0
    while len(pts) < N and tries < max_tries:
        tries += 1
        x = cx + rng.normal() * sx
        y = cy + rng.normal() * sy
        if (x1 <= x <= x2) and (y1 <= y <= y2):
            xi = int(round(x)); yi = int(round(y))
            xi = max(0, min(W - 1, xi))
            yi = max(0, min(H - 1, yi))
            pts.append((xi, yi))

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # 부족하면 bbox 내부 uniform으로 채움
    while len(pts) < N:
        xi = rng.integers(int(x1), int(x2) + 1)
        yi = rng.integers(int(y1), int(y2) + 1)
        pts.append((int(xi), int(yi)))

    return np.array(pts, dtype=np.int32)


def local_median_filter(depth: np.ndarray, xy: np.ndarray, ksize: int, delta: float) -> np.ndarray:
    H, W = depth.shape
    r = ksize // 2
    ok = np.ones((xy.shape[0],), dtype=bool)
    for i, (x, y) in enumerate(xy):
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        patch = depth[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch)]
        if patch.size == 0:
            ok[i] = False
            continue
        m = np.median(patch)
        d = depth[y, x]
        if (not np.isfinite(d)) or abs(d - m) > delta:
            ok[i] = False
    return ok


def project_pixels_to_world(
    xy: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    pose_4x4: np.ndarray,
    pose_is_T_world_cam: bool,
    depth_scale: float
) -> np.ndarray:
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    d = depth[y.astype(np.int32), x.astype(np.int32)].astype(np.float32) * float(depth_scale)

    X = (x - cx) * d / fx
    Y = (y - cy) * d / fy
    Z = d
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # (N,4)

    T = pose_4x4.astype(np.float32)
    if not pose_is_T_world_cam:
        T = np.linalg.inv(T)

    pts_w = (T @ pts_cam.T).T[:, :3]
    return pts_w


def min_mahalanobis_inlier_mask(
    pts_w: np.ndarray,
    obj_gaussians: List[Dict[str, np.ndarray]],
    T_gate: float
) -> np.ndarray:
    if pts_w.shape[0] == 0 or len(obj_gaussians) == 0:
        return np.zeros((pts_w.shape[0],), dtype=bool)

    min_d2 = np.full((pts_w.shape[0],), np.inf, dtype=np.float32)
    for g in obj_gaussians:
        mu = g["mu"]
        invS = g["invS"]
        diff = (pts_w - mu[None, :]).astype(np.float32)
        tmp = diff @ invS
        d2 = np.sum(tmp * diff, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return (min_d2 < float(T_gate))


# =========================
# Coarse-CD per-frame (raw_gobs -> changed?)
# =========================
def coarse_cd_per_frame_from_raw_gobs(
    frame_idx: int,
    raw_gobs: dict,                  # 여기선 gobs(필터+mask_subtract_contained 적용된 것) 넣을 것
    depth_array: np.ndarray,          # (H,W)
    intrinsics_4x4: np.ndarray,       # intrinsics
    pose_4x4: np.ndarray,             # adjusted_pose
    before_objects,
    gauss_cache,
    label_to_obj,
    cfg,
    rng: np.random.Generator,
):
    H, W = depth_array.shape
    K = intrinsics_4x4[:3, :3]
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    xyxy = np.asarray(raw_gobs["xyxy"])
    confs = np.asarray(raw_gobs["confidence"])
    cls_ids = np.asarray(raw_gobs["class_id"]).astype(int)
    masks = raw_gobs.get("mask", None)  # (N,H,W) expected
    classes_arr = raw_gobs.get("classes", None)

    frame_changed = False
    det_logs = []
    changed_classes = []

    for j in range(xyxy.shape[0]):
        bbox = tuple(xyxy[j].tolist())
        conf = float(confs[j])

        if classes_arr is not None and len(classes_arr) > 0:
            cls_name = str(classes_arr[int(cls_ids[j])])
        else:
            cls_name = str(int(cls_ids[j]))

        # A) sampling: mask 내부에서 샘플링 (mask 없거나 비면 bbox 샘플링 fallback)
        if masks is not None and len(masks) > j:
            xy = sample_pixels_from_mask_centerbiased(
                mask=masks[j],
                bbox_xyxy=bbox,
                N=cfg.N,
                sigma_ratio=cfg.sigma,
                rng=rng,
            )
        else:
            xy = center_biased_sample_pixels(
                bbox_xyxy=bbox, H=H, W=W,
                N=cfg.N, sigma_ratio=cfg.sigma,
                rng=rng, max_tries=cfg.max_resample_tries
            )

        if xy.shape[0] == 0:
            continue

        # B) depth validity + (optional) median filter
        d = depth_array[xy[:,1], xy[:,0]]
        valid = np.isfinite(d) & (d > 0)
        xy = xy[valid]
        if xy.shape[0] == 0:
            continue

        if cfg.use_local_median:
            ok = local_median_filter(depth_array, xy, cfg.median_ksize, cfg.delta_depth)
            xy = xy[ok]

        if xy.shape[0] < cfg.min_valid:
            det_logs.append({"det": j, "skip": True, "reason": "min_valid", "cls": cls_name})
            continue

        # C) 2D->3D
        pts_w = project_pixels_to_world(
            xy=xy, depth=depth_array,
            fx=fx, fy=fy, cx=cx, cy=cy,
            pose_4x4=pose_4x4,
            pose_is_T_world_cam=cfg.pose_is_T_world_cam,
            depth_scale=cfg.depth_scale
        )

        # E) label optional gating (tau_conf 사용처는 여기 맞음) :contentReference[oaicite:4]{index=4}
        if conf >= cfg.tau_conf and (cls_name in label_to_obj):
            candidate_objects = label_to_obj[cls_name]
            if len(candidate_objects) == 0:
                candidate_objects = list(range(len(before_objects)))
            used_label = True
        else:
            candidate_objects = list(range(len(before_objects)))
            used_label = False

        # D) inlier test
        inlier_any = np.zeros((pts_w.shape[0],), dtype=bool)
        for obj_idx in candidate_objects:
            inlier_any |= min_mahalanobis_inlier_mask(pts_w, gauss_cache[obj_idx], cfg.T_gate)
            if inlier_any.mean() > 0.95:
                break

        inlier_ratio = float(inlier_any.mean())
        change_candidate = (inlier_ratio < cfg.tau_inlier)

        det_logs.append({
            "det": j,
            "conf": conf,
            "cls": cls_name,
            "used_label": used_label,
            "valid_samples": int(pts_w.shape[0]),
            "inlier_ratio": inlier_ratio,
            "change_candidate": bool(change_candidate),
        })

        # ✅ break 금지: 한 프레임에서 여러 변화 객체를 다 모아야 함
        if change_candidate:
            frame_changed = True
            changed_classes.append(cls_name)

    # 중복 제거(순서 유지)
    uniq = []
    seen = set()
    for c in changed_classes:
        if c not in seen:
            uniq.append(c); seen.add(c)
    changed_classes = uniq

    frame_stat = {
        "frame_idx": frame_idx,
        "num_dets": int(xyxy.shape[0]),
        "changed": bool(frame_changed),
        "dets": det_logs,
    }
    return frame_changed, frame_stat, changed_classes
    # return frame_changed, frame_stat
# New_script_End_Coarse

# New_script_Start
def _sigma_to_rot_scale(Sigma: np.ndarray, eps: float = 1e-6):
    """
    Sigma (3,3) -> rotation matrix R (3,3), scale (3,)
    scale is stddev along principal axes: sqrt(eigvals)
    """
    S = 0.5 * (Sigma + Sigma.T) + np.eye(3) * eps
    evals, evecs = np.linalg.eigh(S)  # asc
    evals = np.maximum(evals, eps)
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1
    scale = np.sqrt(evals)
    R = evecs
    return R, scale

def _rotmat_to_quat_xyzw(R: np.ndarray):
    """Quaternion [x,y,z,w]"""
    m00,m01,m02 = R[0,0],R[0,1],R[0,2]
    m10,m11,m12 = R[1,0],R[1,1],R[1,2]
    m20,m21,m22 = R[2,0],R[2,1],R[2,2]
    tr = m00+m11+m22
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        qw = 0.25*S
        qx = (m21-m12)/S
        qy = (m02-m20)/S
        qz = (m10-m01)/S
    elif (m00>m11) and (m00>m22):
        S = np.sqrt(1.0+m00-m11-m22)*2
        qw = (m21-m12)/S
        qx = 0.25*S
        qy = (m01+m10)/S
        qz = (m02+m20)/S
    elif m11>m22:
        S = np.sqrt(1.0+m11-m00-m22)*2
        qw = (m02-m20)/S
        qx = (m01+m10)/S
        qy = 0.25*S
        qz = (m12+m21)/S
    else:
        S = np.sqrt(1.0+m22-m00-m11)*2
        qw = (m10-m01)/S
        qx = (m02+m20)/S
        qy = (m12+m21)/S
        qz = 0.25*S
    return np.array([qx,qy,qz,qw], dtype=np.float32)

def export_objects_gaussians_and_samples(
    objects,
    out_dir: str,
    gaussians_ply_name: str = "gaussians.ply",
    samples_ply_name: str = "gaussian_samples.ply",
    meta_json_name: str = "gaussians_meta.json",
    clip_npz_name: str = "clip_features.npz",
    scale_multiplier: float = 1.0,
    opacity_default: float = 1.0,
    save_clip_float16: bool = True
):
    """
    Export:
      (A) All obj["gaussians"] -> 3DGS-like .ply (same as before)
      (B) All per-gaussian saved samples -> colored points .ply
      (C) Meta json including (object_index, class_name, gaussian_index) for each gaussian

    Why split?
      - 3DGS PLY format is (mostly) fixed numeric properties; it cannot reliably store strings.
      - samples are best exported as separate point cloud for visualization / debugging.
      - meta json keeps mappings and extra fields (class_name etc.).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_gauss_ply = out_dir / gaussians_ply_name
    out_samples_ply = out_dir / samples_ply_name
    out_meta_json = out_dir / meta_json_name
    out_clip_npz = out_dir / clip_npz_name
    
    # -------------------------
    # (A) 3DGS-like gaussians PLY
    # -------------------------
    rows = []
    meta = []  # json-friendly per-gaussian records
    sample_pts = []
    sample_cols = []
    clip_dict = {}

    for obj_index, obj in enumerate(objects):
        clip = obj.get("clip_ft", None)

        if clip is not None:
            if torch.is_tensor(clip):
                clip_np = clip.detach().cpu().numpy()
            else:
                clip_np = np.asarray(clip)

            clip_np = clip_np.astype(np.float16 if save_clip_float16 else np.float32)
            clip_dict[str(obj_index)] = clip_np
        
        if obj.get("num_detections", 1) < 1:
            continue
        if obj.get("is_background", False):
            continue

        gaussians = obj.get("gaussians", None)
        if not gaussians:
            continue

        for g in gaussians:
            # required fields
            mu = np.asarray(g["mu"], dtype=np.float32)
            Sigma = np.asarray(g["Sigma"], dtype=np.float32)

            # rot/scale
            R, scale = _sigma_to_rot_scale(Sigma)
            scale = scale * float(scale_multiplier)
            quat = _rotmat_to_quat_xyzw(R)

            # gaussian mean color
            rgb01 = np.asarray(g.get("rgb", [1, 1, 1]), dtype=np.float32)
            rgb01 = np.clip(rgb01, 0.0, 1.0)
            f_dc = rgb01  # SH degree 0 only

            opacity = float(g.get("alpha", opacity_default))

            rows.append([
                mu[0], mu[1], mu[2],
                0.0, 0.0, 0.0,               # nx ny nz dummy
                f_dc[0], f_dc[1], f_dc[2],    # f_dc_0..2
                opacity,
                scale[0], scale[1], scale[2],
                quat[0], quat[1], quat[2], quat[3],
            ])

            # -------------------------
            # (C) meta json (keep strings here)
            # -------------------------
            meta.append({
                "object_index": int(g.get("object_index", obj.get("object_index", -1))),
                "class_name": str(g.get("class_name", obj.get("class_name", "unknown"))),
                "gaussian_index": int(g.get("gaussian_index", -1)),
                "w": float(g.get("w", 1.0)),
                "alpha": float(g.get("alpha", opacity_default)),
                "mu": mu.tolist(),
                "Sigma": np.asarray(Sigma, dtype=np.float32).tolist(),
                "rgb": rgb01.tolist(),
            })

            # -------------------------
            # (B) samples -> colored point cloud PLY
            # -------------------------
            samples = g.get("samples", None)
            if samples:
                # xyz: stored as float16 list, rgb: uint8 list (default) or float list
                xyz = np.asarray([s["xyz"] for s in samples], dtype=np.float32)  # (M,3)
                rgb = np.asarray([s["rgb"] for s in samples])
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.float32)
                    if np.nanmax(rgb) <= 1.5:  # assume 0..1
                        rgb = rgb * 255.0
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

                # store
                sample_pts.append(xyz)
                sample_cols.append(rgb)

    if len(clip_dict) > 0:
        np.savez_compressed(out_clip_npz, **clip_dict)
    
    # write gaussians ply
    rows = np.asarray(rows, dtype=np.float32)
    if rows.shape[0] == 0:
        print("[WARN] No gaussians found to export.")
        return

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {rows.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ])

    with open(out_gauss_ply, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(map(lambda x: f"{x:.6f}", r.tolist())) + "\n")

    # write samples ply (colored points)
    if len(sample_pts) > 0:
        P = np.concatenate(sample_pts, axis=0).astype(np.float32)
        C = np.concatenate(sample_cols, axis=0).astype(np.uint8)
        assert P.shape[0] == C.shape[0]

        header_s = "\n".join([
            "ply",
            "format ascii 1.0",
            f"element vertex {P.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ])
        with open(out_samples_ply, "w") as f:
            f.write(header_s + "\n")
            for i in range(P.shape[0]):
                x, y, z = P[i].tolist()
                r, g, b = C[i].tolist()
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    else:
        # still create an empty file? keep it simple: just warn
        print("[WARN] No per-gaussian samples found to export (samples_ply not written).")

    # write meta json
    try:
        import json
        with open(out_meta_json, "w") as f:
            json.dump(
                {
                    "num_gaussians": int(rows.shape[0]),
                    "has_samples": bool(len(sample_pts) > 0),
                    "gaussians": meta,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        print(f"[WARN] Failed to write meta json: {e}")

    print(f"[INFO] Exported {rows.shape[0]} gaussians to: {out_gauss_ply}")
    if len(sample_pts) > 0:
        print(f"[INFO] Exported {P.shape[0]} sample points to: {out_samples_ply}")
    print(f"[INFO] Exported meta json to: {out_meta_json}")
# New_script_End

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
    tracker = MappingTracker()
    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()
    
    # New_script_Start
    # before_objects는 네가 이미 로드한다고 했으니 그대로 사용
    # (초기 3D scene의 gaussian + 3D RGB sample points가 들어있는 representation)
    before_rep_dir = Path("/home/pchy0316/dataset/my_local_data/Replica/room0/3D_representation")
    before_objects = load_before_objects(str(before_rep_dir))

    # NEW: 2D mask center 기반 change detection 설정 + cache (1회만 빌드)
    cfg_cd = ClipCenterCDCfg(
        clip_topk=5,
        tau_clip_sim=0.25,
        tau_new_gaussian_dist=0.25,
        use_det_conf_gate_for_created=True,
        tau_conf_created=0.6,
        tau_match_gaussian_dist=0.25,
        use_det_conf_gate_for_match=False,
        tau_conf_match=0.0,
        depth_scale=1.0,
        center_depth_ksize=5,
        min_depth=1e-6,
    )
    before_cache = build_before_scene_clip_gaussian_cache(before_objects, eps=cfg_cd.eps)


    # Disappeared 판정용: 시퀀스 전체에서 관측된 object_index 표시
    alive_flags = np.zeros((len(before_cache.obj_class),), dtype=bool)

    frame_to_created: Dict[int, List[str]] = {}
    frame_cd_stats: Dict[int, Dict[str, Any]] = {}





    # # 디버깅 출력용
    # obj0 = before_objects[0]
    # print("디버깅디버깅디버깅시작")
    # print(obj0["class_name"])
    # print(len(obj0["gaussians"]))
    # print(obj0["object_meta"]["bbox_center"])
    # print("디버깅디버깅디버깅끝남")
    
    # New_script_Start

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # NEW: created objects (sequence-level 3D objects from per-frame created masks)
    created_objects = MapObjectList(device=cfg.device)
    created_map_edges = MapEdgeMapping(created_objects)

    # NEW: created 3D object confirmation (연속 프레임 검출 기반 노이즈 억제)
    # - created_objects에 한두 프레임만 튀는 노이즈가 쌓이는 것을 막기 위해,
    #   시퀀스 끝에서 '연속 N프레임 이상 관측된 객체만' 남긴다.
    cfg_created_confirm = Created3DConfirmCfg(
        min_consecutive_frames=5,
        min_total_detections=10,
    )

    def _cd_track_init(obj: Dict[str, Any], fi: int):
        fi = int(fi)
        if "_cd_track_id" not in obj:
            obj["_cd_track_id"] = uuid.uuid4().hex
        obj["_cd_last_frame"] = fi
        obj["_cd_curr_run"] = 1
        obj["_cd_max_run"] = 1
        obj["_cd_frames"] = [fi]

    def _cd_track_hit(obj: Dict[str, Any], fi: int):
        """이 created object가 현재 프레임에서 관측되었음을 기록 (프레임 단위로만 1회 카운트)."""
        fi = int(fi)
        last = obj.get("_cd_last_frame", None)
        # 같은 프레임에서 여러 detection이 같은 object로 붙을 수 있으므로 중복 카운트 방지
        if last is not None and int(last) == fi:
            return

        if last is not None and int(last) == fi - 1:
            curr = int(obj.get("_cd_curr_run", 1)) + 1
        else:
            curr = 1

        obj["_cd_curr_run"] = curr
        obj["_cd_max_run"] = max(int(obj.get("_cd_max_run", 1)), curr)
        obj["_cd_last_frame"] = fi

        frames = obj.get("_cd_frames", None)
        if not isinstance(frames, list):
            frames = []
            obj["_cd_frames"] = frames
        frames.append(fi)

    def _cd_track_snapshot(objs: Any) -> List[Dict[str, Any]]:
        snap = []
        for o in objs:
            snap.append({
                "_cd_track_id": o.get("_cd_track_id", None),
                "_cd_last_frame": o.get("_cd_last_frame", None),
                "_cd_curr_run": o.get("_cd_curr_run", None),
                "_cd_max_run": o.get("_cd_max_run", None),
                "_cd_frames": o.get("_cd_frames", None),
            })
        return snap

    def _cd_track_restore_by_index(objs: Any, snap: List[Dict[str, Any]]):
        # merge_obj_matches가 내부 dict를 재구성하는 경우 대비 (order가 유지된다는 가정 하에 인덱스로 복구)
        n = int(min(len(objs), len(snap)))
        for i in range(n):
            s = snap[i]
            if s is None:
                continue
            o = objs[i]
            if "_cd_track_id" not in o and s.get("_cd_track_id") is not None:
                o["_cd_track_id"] = s["_cd_track_id"]
            if "_cd_last_frame" not in o and s.get("_cd_last_frame") is not None:
                o["_cd_last_frame"] = s["_cd_last_frame"]
            if "_cd_curr_run" not in o and s.get("_cd_curr_run") is not None:
                o["_cd_curr_run"] = s["_cd_curr_run"]
            if "_cd_max_run" not in o and s.get("_cd_max_run") is not None:
                o["_cd_max_run"] = s["_cd_max_run"]
            if "_cd_frames" not in o and isinstance(s.get("_cd_frames"), list):
                o["_cd_frames"] = list(s["_cd_frames"])

    def _max_consecutive_from_frames(frames: List[int]) -> int:
        if not isinstance(frames, list) or len(frames) == 0:
            return 0
        fr = sorted(set(int(x) for x in frames))
        best = 1
        run = 1
        for a, b in zip(fr, fr[1:]):
            if b == a + 1:
                run += 1
            else:
                best = max(best, run)
                run = 1
        best = max(best, run)
        return int(best)

    def _prune_created_objects_by_consecutive(objs: Any, confirm_cfg: Created3DConfirmCfg):
        if objs is None or len(objs) == 0:
            return objs

        kept = []
        for o in objs:
            # prefer tracked max run if exists
            max_run = o.get("_cd_max_run", None)
            if max_run is None:
                max_run = _max_consecutive_from_frames(o.get("_cd_frames", []))
                if max_run == 0:
                    max_run = int(o.get("num_detections", 0) or 0)

            total = o.get("num_detections", None)
            if total is None:
                total = len(o.get("_cd_frames", [])) if isinstance(o.get("_cd_frames", []), list) else 0
            total = int(total)

            if (int(max_run) >= int(confirm_cfg.min_consecutive_frames)) and (total >= int(confirm_cfg.min_total_detections)):
                kept.append(o)

        out = MapObjectList(device=cfg.device)
        out.extend(kept)
        return out


    # For visualization
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        sam_predictor = SAM('sam_l.pt') # SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    openai_client = get_openai_client()

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)): # len(dataset) --> 1500
        tracker.curr_frame_idx = frame_idx
        counter+=1        
        orr.set_time_sequence("frame", frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()


            # 이 sam은 여기서 감지용으로 사용하지만 추후에 사용하지
            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            # Make the edges # 이부분 GPT 연동해서 edge 생성하는 부분
            # labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, openai_client)

            labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, False, None)
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
            }

            raw_gobs = results

            # # save the detections if needed
            # if cfg.save_detections:

            #     vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
            #     # Visualize and save the annotated image
            #     annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
            #     cv2.imwrite(str(vis_save_path), annotated_image)

            #     depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
            #     depth_image_rgb = depth_image_rgb.astype(np.uint8)
            #     depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
            #     annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
            #     cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
            #     cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
            #     save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}.")

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
        
        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor)
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        # ---------------------------------
        # NEW: change detection (2D mask center 기반)
        # ---------------------------------
        created_labels, created_det_indices, matched_obj_indices, cd_stat = clip_center_cd_per_frame(
            frame_idx=frame_idx,
            gobs=gobs,
            depth_array=depth_array,
            intrinsics_4x4=intrinsics.cpu().numpy(),
            pose_4x4=adjusted_pose,
            before_cache=before_cache,
            cfg=cfg_cd,
        )
        # Disappeared 판정용: 이 프레임에서 기존 object로 매칭된 object_index는 alive로 표시
        if matched_obj_indices is not None and len(matched_obj_indices) > 0:
            try:
                alive_flags[np.asarray(matched_obj_indices, dtype=np.int32)] = True
            except Exception:
                for oi in matched_obj_indices:
                    if 0 <= int(oi) < len(alive_flags):
                        alive_flags[int(oi)] = True

        frame_to_created[frame_idx] = created_labels
        frame_cd_stats[frame_idx] = cd_stat

        # (선택) 콘솔에 바로 보고 싶으면
        # if len(changed_labels) > 0:
        #     print(f"[CD] frame {frame_idx}: {changed_labels}")


        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
            continue


        # ---------------------------------
        # NEW: created mask들을 3D object로 누적/병합 (시퀀스 전체 created 3D objects)
        # ---------------------------------
        if created_det_indices is not None and len(created_det_indices) > 0:
            try:
                # gobs와 obj_pcds_and_bboxes를 created det index로 슬라이싱
                gobs_created = _slice_gobs_by_indices(gobs, created_det_indices)
                obj_pcds_and_bboxes_created = [obj_pcds_and_bboxes[int(ii)] for ii in created_det_indices]

                created_detection_list = make_detection_list_from_pcd_and_gobs(
                    obj_pcds_and_bboxes_created, gobs_created, color_path, obj_classes, frame_idx
                )

                # 유효 detection만 남으면 merge 수행
                if len(created_detection_list) > 0:
                    if len(created_objects) == 0:
                        created_objects.extend(created_detection_list)
                        # init track meta for consecutive-frame confirmation
                        for _o in created_objects:
                            try:
                                _cd_track_init(_o, frame_idx)
                            except Exception:
                                pass
                    else:
                        spatial_sim_c = compute_spatial_similarities(
                            spatial_sim_type=cfg['spatial_sim_type'],
                            detection_list=created_detection_list,
                            objects=created_objects,
                            downsample_voxel_size=cfg['downsample_voxel_size']
                        )
                        visual_sim_c = compute_visual_similarities(created_detection_list, created_objects)
                        agg_sim_c = aggregate_similarities(
                            match_method=cfg['match_method'],
                            phys_bias=cfg['phys_bias'],
                            spatial_sim=spatial_sim_c,
                            visual_sim=visual_sim_c
                        )
                        match_indices_c = match_detections_to_objects(
                            agg_sim=agg_sim_c,
                            detection_threshold=cfg['sim_threshold']
                        )

                        # NEW: consecutive-frame confirmation용 track 업데이트 (matched object만 frame hit)
                        try:
                            for _det_i, _oi in enumerate(match_indices_c):
                                if _oi is None:
                                    continue
                                _oi = int(_oi)
                                if _oi >= 0 and _oi < len(created_objects):
                                    _cd_track_hit(created_objects[_oi], frame_idx)
                        except Exception:
                            pass

                        # NEW: merge 과정에서 dict가 재구성될 수 있어 track 필드 스냅샷/복구(best-effort)
                        _track_snap = _cd_track_snapshot(created_objects)

                        created_objects = merge_obj_matches(
                            detection_list=created_detection_list,
                            objects=created_objects,
                            match_indices=match_indices_c,
                            downsample_voxel_size=cfg['downsample_voxel_size'],
                            dbscan_remove_noise=cfg['dbscan_remove_noise'],
                            dbscan_eps=cfg['dbscan_eps'],
                            dbscan_min_points=cfg['dbscan_min_points'],
                            spatial_sim_type=cfg['spatial_sim_type'],
                            device=cfg['device']
                        )

                        # NEW: track 필드 복구 + 새 객체 초기화
                        try:
                            _cd_track_restore_by_index(created_objects, _track_snap)
                        except Exception:
                            pass
                        for _o in created_objects:
                            if "_cd_track_id" not in _o:
                                try:
                                    _cd_track_init(_o, frame_idx)
                                except Exception:
                                    pass

                    # created_objects class_name 안정화 (가장 많이 나온 class_id로)
                    for _idx, _obj in enumerate(created_objects):
                        try:
                            curr_cnt = Counter(_obj.get('class_id', []))
                            if len(curr_cnt) > 0:
                                most_common_class_id = curr_cnt.most_common(1)[0][0]
                                _obj['class_name'] = obj_classes.get_classes_arr()[int(most_common_class_id)]
                        except Exception:
                            pass

                    # edge 구조는 여기서는 필요 없지만, 일관성을 위해 최신화
                    created_map_edges = MapEdgeMapping(created_objects)

            except Exception as e:
                print(f"[WARN] failed to build/merge created 3D objects at frame {frame_idx}: {e}")

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
            continue 

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        # fix the class names for objects
        # they should be the most popular name, not the first name
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Clean up outlier edges
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            obj1_class_name = objects[curr_obj1_idx]['class_name'] 
            obj2_class_name = objects[curr_obj2_idx]['class_name']
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges
            )
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )
        
        if cfg.vis_render:
            # render a frame, if needed (not really used anymore since rerun)
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )

        # if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
        #     # save the pointcloud
        #     save_pointcloud(
        #         exp_suffix=cfg.exp_suffix,
        #         exp_out_path=exp_out_path,
        #         cfg=cfg,
        #         objects=objects,
        #         obj_classes=obj_classes,
        #         latest_pcd_filepath=cfg.latest_pcd_filepath,
        #         create_symlink=True
        #     )

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
    # LOOP OVER -----------------------------------------------------

    # # New_script_Start_Coarse
    # output_text = f"changed_frames: {changed_frame_indices}"
    
    # print("결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다")
    # print(output_text)
    # print("결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다")
    # out_txt = exp_out_path / "coarse_changed_frames.txt"
    # with open(out_txt, "w") as f:
    #     f.write(output_text + "\n")

    # # (선택) debug json
    # out_json = exp_out_path / "coarse_changed_frames_debug.json"
    # with open(out_json, "w") as f:
    #     json.dump(per_frame_stats, f, indent=2, ensure_ascii=False)
        
    # # 기본 changed frame index 출력
    # output_text = f"changed_frames: {changed_frame_indices}"
    # print("결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다")
    # print(output_text)
    # print("결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다결과입니다 결과입니다 coarse change detection 결과입니다")

    # with open(exp_out_path / "coarse_changed_frames.txt", "w") as f:
    #     f.write(output_text + "\n")

    # # 🔥 클래스 포함 txt 출력
    # class_txt_path = exp_out_path / "coarse_changed_frame_classes.txt"
    # with open(class_txt_path, "w") as f:
    #     for frame_idx in sorted(changed_frame_classes.keys()):
    #         class_list = changed_frame_classes[frame_idx]
    #         line = f"frame {frame_idx}: {', '.join(class_list)}\n"
    #         f.write(line)

    # print(f"[INFO] saved: {class_txt_path}")

    # ------------------------------
    # Change detection 결과 저장
    # ------------------------------
    # 요구사항: "frame: 클래스 라벨들" 형태로 프레임별 변화 클래스 출력
# ------------------------------
    # Disappeared 결과(시퀀스 전체) 저장
    # ------------------------------
    # 요구사항: 이미지 시퀀스 동안 한 번도 매칭되지 않은 before object_index를 시퀀스 끝나고 출력
    # (요구사항 0) 배경(벽/천장/바닥 등) 클래스는 disappeared 출력에서도 제외
    _bg_set_cd = set()
    try:
        _bg_set_cd = {str(x).strip().lower() for x in (cfg_cd.bg_class_names or ())}
    except Exception:
        _bg_set_cd = set()

    disappeared_internal_indices = []
    for i, alive in enumerate(alive_flags.tolist()):
        if bool(alive):
            continue
        try:
            cls_i = str(before_cache.obj_class[int(i)]).strip().lower()
        except Exception:
            cls_i = ""
        if cls_i in _bg_set_cd:
            continue
        disappeared_internal_indices.append(int(i))


    # gaussians_meta의 object_index(=obj_ids) 기준으로 출력
    disappeared_object_indices = [int(before_cache.obj_ids[i]) for i in disappeared_internal_indices if 0 <= int(i) < len(before_cache.obj_ids)]

    out_dis_path = exp_out_path / "disappeared_object_indices.txt"
    with open(out_dis_path, "w") as f:
        for oi in disappeared_object_indices:
            f.write(f"{oi}\n")
    print(f"[INFO] saved disappeared object indices -> {out_dis_path}")

    # (옵션) class_name도 같이 저장 (디버깅/가독성)
    out_dis2_path = exp_out_path / "disappeared_objects_with_class.txt"
    with open(out_dis2_path, "w") as f:
        for internal_i in disappeared_internal_indices:
            if 0 <= int(internal_i) < len(before_cache.obj_ids):
                ext_id = int(before_cache.obj_ids[int(internal_i)])
                cls = str(before_cache.obj_class[int(internal_i)])
                f.write(f"{ext_id}: {cls}\n")

    print(f"[INFO] saved disappeared objects(with class) -> {out_dis2_path}")
    # ------------------------------
    # Created 결과 저장
    # ------------------------------
    out_created_txt_path = exp_out_path / "frame_created_labels.txt"
    with open(out_created_txt_path, "w") as f:
        for fi in sorted(frame_to_created.keys()):
            labels = frame_to_created[fi]
            if labels is None:
                labels = []
            f.write(f"{fi}: {', '.join(labels)}\n")
    print(f"[INFO] saved per-frame created labels -> {out_created_txt_path}")


    # (선택) 사라짐/생김을 분리해서 보고 싶으면 디버그 json 저장
    out_json_path = exp_out_path / "frame_change_debug.json"
    try:
        debug_payload = {
            "disappeared_object_indices": disappeared_object_indices,
            "disappeared_internal_indices": disappeared_internal_indices,
            "alive_flags": alive_flags.tolist(),
            "per_frame": frame_cd_stats,
        }
        with open(out_json_path, "w") as f:
            json.dump(debug_payload, f, indent=2, ensure_ascii=False)
        print(f"[INFO] saved change debug json -> {out_json_path}")
    except Exception as e:
        print(f"[WARN] failed to save change debug json: {e}")
    # New_script_End_CD


    # ------------------------------
    # NEW: 시퀀스 전체 생성된 3D 객체(created_objects) 저장
    # ------------------------------
    try:
        created_out_dir = exp_out_path / "created_3d_objects"
        created_out_dir.mkdir(parents=True, exist_ok=True)

        # (선택) created_objects에 대해 1회 추가 정제/병합
        if len(created_objects) > 0:
            # NEW: 연속 프레임 등장 조건으로 created_objects 1차 정제 (노이즈 제거)
            try:
                created_objects = _prune_created_objects_by_consecutive(created_objects, cfg_created_confirm)
                created_map_edges = MapEdgeMapping(created_objects)
            except Exception as _e:
                print(f"[WARN] created_objects consecutive pruning failed: {_e}")

            try:
                created_objects = denoise_objects(
                    downsample_voxel_size=cfg['downsample_voxel_size'],
                    dbscan_remove_noise=cfg['dbscan_remove_noise'],
                    dbscan_eps=cfg['dbscan_eps'],
                    dbscan_min_points=cfg['dbscan_min_points'],
                    spatial_sim_type=cfg['spatial_sim_type'],
                    device=cfg['device'],
                    objects=created_objects
                )
            except Exception as _e:
                print(f"[WARN] created_objects denoise failed: {_e}")

            try:
                created_objects = filter_objects(
                    obj_min_points=cfg['obj_min_points'],
                    obj_min_detections=cfg['obj_min_detections'],
                    objects=created_objects,
                    map_edges=created_map_edges
                )
            except Exception as _e:
                print(f"[WARN] created_objects filter failed: {_e}")

            try:
                created_objects, created_map_edges = merge_objects(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=created_objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=False,
                    map_edges=created_map_edges
                )
            except Exception as _e:
                print(f"[WARN] created_objects merge failed: {_e}")

        summary = []
        for ci, obj in enumerate(created_objects):
            pcd = obj.get("pcd", None)
            if pcd is None:
                continue

            ply_name = f"created_obj_{ci:03d}.ply"
            ply_path = created_out_dir / ply_name
            try:
                o3d.io.write_point_cloud(str(ply_path), pcd)
            except Exception as _e:
                print(f"[WARN] failed to write created obj ply ({ci}): {_e}")

            # centroid
            center = None
            try:
                bbox = obj.get("bbox", None)
                if bbox is not None:
                    center = np.asarray(bbox.get_center()).astype(np.float32).tolist()
            except Exception:
                center = None
            if center is None:
                try:
                    pts = np.asarray(pcd.points)
                    if pts.size > 0:
                        center = pts.mean(axis=0).astype(np.float32).tolist()
                except Exception:
                    center = None

            summary.append({
                "created_obj_id": int(ci),
                "class_name": str(obj.get("class_name", "unknown")),
                "centroid": center,
                "num_points": int(len(pcd.points)),
                "num_detections": int(obj.get("num_detections", 0)) if isinstance(obj.get("num_detections", 0), (int, float)) else None,
                "ply": ply_name,
            })

        with open(created_out_dir / "created_objects_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[INFO] saved created 3D objects -> {created_out_dir} (n={len(summary)})")
    except Exception as e:
        print(f"[WARN] failed to export created 3D objects: {e}")



    # Paste this under:  # LOOP OVER -----------------------------------------------
    # =========================

    # --- Optional: try sklearn KMeans, fallback to torch if not available ---
    _USE_SKLEARN = True
    try:
        from sklearn.cluster import KMeans
    except Exception:
        _USE_SKLEARN = False

    def _torch_kmeans(X: np.ndarray, K: int, iters: int = 30, seed: int = 0) -> np.ndarray:
        """
        Simple torch k-means fallback (no sklearn).
        Returns labels (N,)
        """
        import torch
        torch.manual_seed(seed)

        x = torch.from_numpy(X).float().cuda() if torch.cuda.is_available() else torch.from_numpy(X).float()
        N = x.shape[0]
        # init centers from random points
        idx = torch.randperm(N)[:K]
        centers = x[idx].clone()

        for _ in range(iters):
            # assign
            d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)  # (N,K)
            labels = torch.argmin(d2, dim=1)  # (N,)
            # update
            new_centers = []
            for k in range(K):
                mask = (labels == k)
                if mask.any():
                    new_centers.append(x[mask].mean(dim=0))
                else:
                    # re-init empty cluster
                    new_centers.append(x[torch.randint(0, N, (1,)).item()])
            new_centers = torch.stack(new_centers, dim=0)
            # stop if converged
            if torch.allclose(new_centers, centers, atol=1e-4, rtol=0):
                centers = new_centers
                break
            centers = new_centers

        return labels.detach().cpu().numpy()

    # -------------------------
    # NEW: robust sampling utils
    # -------------------------
    def _filter_valid_points_basic(
        X: np.ndarray,
        C: np.ndarray | None,
        nn_min: int = 5,
        nn_radius: float = 0.03,
        rgb_min_norm: float = 0.05,
        rgb_white_eps: float = 0.02,
    ):
        """
        Basic defense:
        1) RGB validity filter (remove near-black / near-white / NaN)
        2) Local density filter using radius neighbors (remove floating/empty-space points)
        Returns: valid_mask (N,)
        """
        N = X.shape[0]
        valid = np.ones((N,), dtype=bool)

        # RGB validity
        if C is not None:
            bad = np.any(~np.isfinite(C), axis=1)
            rgb_norm = np.linalg.norm(C, axis=1)          # 0~1 scale => 0~1.732
            too_dark = rgb_norm < rgb_min_norm
            too_white = np.linalg.norm(C - 1.0, axis=1) < rgb_white_eps
            valid &= ~(bad | too_dark | too_white)

        # Local density (Open3D KDTree)
        try:
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(X.astype(np.float64))
            kdt = o3d.geometry.KDTreeFlann(pcd_tmp)
            for i in range(N):
                if not valid[i]:
                    continue
                _, idx, _ = kdt.search_radius_vector_3d(X[i].astype(np.float64), float(nn_radius))
                # idx includes itself
                if (len(idx) - 1) < int(nn_min):
                    valid[i] = False
        except Exception:
            # If KDTree fails, just skip density filtering
            pass

        return valid

    def _robust_sample_rgb_points_for_cluster(
        X: np.ndarray,            # (Nc,3)
        C: np.ndarray | None,     # (Nc,3) in 0~1
        mu: np.ndarray,           # (3,)
        M: int = 32,
        surface_quantile: float = 0.7,  # prefer outer points (surface-like)
        seed: int = 0,
        # defense knobs
        max_tries: int = 5,
        nn_min: int = 5,
        nn_radius: float = 0.03,
        store_rgb_uint8: bool = True,
    ):
        """
        Returns:
        samples(list[dict]): [{"xyz":[...], "rgb":[...]} ...]
        - xyz stored as float16 list
        - rgb stored as uint8 [0..255] list (default) OR float16 [0..1]
        Strategy:
        1) filter invalid RGB + low-density points
        2) sample from "surface candidates" (far from mu) with distance-weighted sampling
        3) if sampling yields issues, relax quantile and retry
        4) fallback to random if needed
        """
        rng = np.random.default_rng(seed)

        if X is None or X.shape[0] == 0:
            return []

        # 1) filter
        valid = _filter_valid_points_basic(X, C, nn_min=nn_min, nn_radius=nn_radius)
        Xv = X[valid]
        Cv = C[valid] if (C is not None) else None

        # fallback if everything got filtered
        if Xv.shape[0] == 0:
            Xv, Cv = X, C

        # helper to pack samples
        def _pack(xyz: np.ndarray, rgb: np.ndarray | None):
            xyz_store = xyz.astype(np.float16)
            if rgb is None:
                rgb_store = (
                    np.full((xyz_store.shape[0], 3), 255, dtype=np.uint8)
                    if store_rgb_uint8 else
                    np.ones((xyz_store.shape[0], 3), dtype=np.float16)
                )
            else:
                if store_rgb_uint8:
                    rgb_store = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                else:
                    rgb_store = np.clip(rgb, 0.0, 1.0).astype(np.float16)
            return [{"xyz": xyz_store[i].tolist(), "rgb": rgb_store[i].tolist()} for i in range(xyz_store.shape[0])]

        # 2) try surface-biased sampling with retries
        q = float(surface_quantile)
        for _ in range(int(max_tries)):
            d = np.linalg.norm(Xv - mu[None, :], axis=1)  # (Nv,)
            thr = np.quantile(d, q) if Xv.shape[0] >= 10 else d.min()

            cand = np.where(d >= thr)[0]
            if cand.size == 0:
                cand = np.arange(Xv.shape[0])

            # distance-weight
            w = d[cand].astype(np.float64)
            w = w - w.min()
            w = w + 1e-12
            w = w / w.sum()

            replace = cand.size < M
            sel = rng.choice(cand, size=M, replace=replace, p=w)

            sx = Xv[sel]
            sc = (Cv[sel] if Cv is not None else None)

            # sanity checks
            if np.all(np.isfinite(sx)) and (sc is None or np.all(np.isfinite(sc))):
                return _pack(sx, sc)

            # relax to include more candidates
            q = max(0.0, q - 0.15)

        # 3) final fallback: random from filtered set
        ridx = rng.choice(Xv.shape[0], size=min(M, Xv.shape[0]), replace=(Xv.shape[0] < M))
        sx = Xv[ridx]
        sc = (Cv[ridx] if Cv is not None else None)
        return _pack(sx, sc)

    def _compute_gaussians_from_pcd(
        pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.02,
        max_points_fit: int = 5000,
        k_min: int = 5,
        k_max: int = 30,
        pts_per_gaussian: int = 300,
        min_cluster_pts: int = 20,
        reg_covar: float = 1e-6,
        seed: int = 0,
        # NEW knobs
        samples_per_gaussian: int = 32,
        surface_quantile: float = 0.7,
        nn_min: int = 5,
        nn_radius: float = 0.03,
        store_rgb_uint8: bool = True,
    ):
        """
        Returns: gaussians(list[dict]), used_points(int)
        Each gaussian dict:
        {
            mu(3), Sigma(3x3), rgb(3), w, alpha,
            samples: list[{"xyz":[x,y,z], "rgb":[r,g,b]}]
        }
        NOTE: obj_index / class_name will be injected OUTSIDE (per-object loop),
            because _compute_gaussians_from_pcd is object-agnostic.
        """
        if pcd is None or len(pcd.points) < 30:
            return [], 0

        # 1) light cleanup
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd_ds.points) < 30:
            return [], 0

        # Optional outlier removal (comment out if too slow)
        try:
            pcd_ds, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        except Exception:
            pass

        pts = np.asarray(pcd_ds.points)  # (N,3)
        if pts.shape[0] < 30:
            return [], 0

        cols = None
        if hasattr(pcd_ds, "colors") and len(pcd_ds.colors) > 0:
            cols = np.asarray(pcd_ds.colors)  # (N,3), 0~1

        # 2) sample for fitting
        N = pts.shape[0]
        if N > max_points_fit:
            rng = np.random.default_rng(seed)
            idx = rng.choice(N, size=max_points_fit, replace=False)
            pts_fit = pts[idx]
            cols_fit = cols[idx] if cols is not None else None
        else:
            pts_fit = pts
            cols_fit = cols

        Nf = pts_fit.shape[0]

        # 3) choose K automatically
        if Nf < min_cluster_pts:
            K = 1
        else:
            K = int(np.clip(round(Nf / pts_per_gaussian), k_min, k_max))
            K = max(1, min(K, Nf // min_cluster_pts))  # prevent too many clusters

        # -----------------
        # K == 1 special case
        # -----------------
        if K == 1:
            mu = pts_fit.mean(axis=0)
            Xm = pts_fit - mu
            Sigma = (Xm.T @ Xm) / max(len(pts_fit) - 1, 1)
            Sigma = Sigma + np.eye(3) * reg_covar
            rgb = cols_fit.mean(axis=0) if cols_fit is not None else np.array([1.0, 1.0, 1.0])

            samples = _robust_sample_rgb_points_for_cluster(
                X=pts_fit, C=cols_fit, mu=mu,
                M=samples_per_gaussian,
                surface_quantile=surface_quantile,
                seed=seed,
                nn_min=nn_min,
                nn_radius=nn_radius,
                store_rgb_uint8=store_rgb_uint8,
            )

            return [{
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "rgb": rgb.tolist(),
                "w": 1.0,
                "alpha": 1.0,
                "samples": samples,
            }], Nf

        # 4) cluster points -> labels
        if _USE_SKLEARN:
            km = KMeans(n_clusters=K, n_init="auto", random_state=seed)
            labels = km.fit_predict(pts_fit)
        else:
            labels = _torch_kmeans(pts_fit, K=K, iters=30, seed=seed)

        # 5) compute Gaussian per cluster + sample rgb points per cluster
        gaussians = []
        for k in range(K):
            idx = np.where(labels == k)[0]
            if idx.size < min_cluster_pts:
                continue

            X = pts_fit[idx]
            mu = X.mean(axis=0)
            Xm = X - mu
            Sigma = (Xm.T @ Xm) / max(len(X) - 1, 1)
            Sigma = Sigma + np.eye(3) * reg_covar

            Ck = cols_fit[idx] if cols_fit is not None else None
            rgb = Ck.mean(axis=0) if Ck is not None else np.array([1.0, 1.0, 1.0])
            w = float(idx.size / Nf)

            samples = _robust_sample_rgb_points_for_cluster(
                X=X, C=Ck, mu=mu,
                M=samples_per_gaussian,
                surface_quantile=surface_quantile,
                seed=seed + 7919 * k,
                nn_min=nn_min,
                nn_radius=nn_radius,
                store_rgb_uint8=store_rgb_uint8,
            )

            gaussians.append({
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "rgb": rgb.tolist(),
                "w": w,
                "alpha": 1.0,
                "samples": samples,
            })

        # Edge-case: if all clusters got pruned
        if len(gaussians) == 0:
            mu = pts_fit.mean(axis=0)
            Xm = pts_fit - mu
            Sigma = (Xm.T @ Xm) / max(len(pts_fit) - 1, 1)
            Sigma = Sigma + np.eye(3) * reg_covar
            rgb = cols_fit.mean(axis=0) if cols_fit is not None else np.array([1.0, 1.0, 1.0])

            samples = _robust_sample_rgb_points_for_cluster(
                X=pts_fit, C=cols_fit, mu=mu,
                M=samples_per_gaussian,
                surface_quantile=surface_quantile,
                seed=seed,
                nn_min=nn_min,
                nn_radius=nn_radius,
                store_rgb_uint8=store_rgb_uint8,
            )

            gaussians = [{
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "rgb": rgb.tolist(),
                "w": 1.0,
                "alpha": 1.0,
                "samples": samples,
            }]

        return gaussians, Nf


    # ---- Run once after loop: build gaussians for each object ----
    # You can tweak these knobs:
    _GM_VOXEL = 0.02         # meters (1~3cm)
    _GM_MAXPTS = 6000        # fit sample cap
    _GM_KMIN = 5
    _GM_KMAX = 30
    _GM_PTS_PER = 300        # larger => fewer gaussians (more coarse)
    _GM_MINC = 20
    _GM_REG = 1e-6

    # sampling knobs
    _GM_SAMPLES_PER = 32     # per-gaussian representative points
    _GM_SURF_Q = 0.7         # 0.6~0.8 (higher => more outer/surface)
    _GM_NN_MIN = 5
    _GM_NN_RADIUS = 0.03     # meters (2~5cm)
    _GM_RGB_UINT8 = True     # compact storage

    # IMPORTANT: loop index(obj_index) is used as "object index"
    for obj_index, obj in enumerate(objects):
        # skip junk/background if your obj dict has these flags
        if obj.get("num_detections", 1) < 1:
            continue
        if obj.get("is_background", False):
            continue
        if "pcd" not in obj or obj["pcd"] is None:
            continue

        gaussians, usedN = _compute_gaussians_from_pcd(
            obj["pcd"],
            voxel_size=_GM_VOXEL,
            max_points_fit=_GM_MAXPTS,
            k_min=_GM_KMIN,
            k_max=_GM_KMAX,
            pts_per_gaussian=_GM_PTS_PER,
            min_cluster_pts=_GM_MINC,
            reg_covar=_GM_REG,
            seed=0,
            samples_per_gaussian=_GM_SAMPLES_PER,
            surface_quantile=_GM_SURF_Q,
            nn_min=_GM_NN_MIN,
            nn_radius=_GM_NN_RADIUS,
            store_rgb_uint8=_GM_RGB_UINT8,
        )

        # NEW: inject object index + class_name into each gaussian
        _cls_name = str(obj.get("class_name", "unknown"))
        for g_i, g in enumerate(gaussians):
            g["object_index"] = int(obj_index)  # <- loop index
            g["class_name"] = _cls_name         # <- from obj["class_name"]
            g["gaussian_index"] = int(g_i)      # (optional but useful) index within this object

        obj["gaussians"] = gaussians
        obj["gaussian_meta"] = {
            "used_points": int(usedN),
            "voxel_size": float(_GM_VOXEL),
            "k_min": int(_GM_KMIN),
            "k_max": int(_GM_KMAX),
            "pts_per_gaussian": int(_GM_PTS_PER),
            "samples_per_gaussian": int(_GM_SAMPLES_PER),
            "surface_quantile": float(_GM_SURF_Q),
            "nn_min": int(_GM_NN_MIN),
            "nn_radius": float(_GM_NN_RADIUS),
            "rgb_uint8": bool(_GM_RGB_UINT8),
        }

    print("[INFO] Built coarse Gaussian mixtures for objects (obj['gaussians']) with per-gaussian RGB samples + labels.")
    # New_script_End New_gaausian


# # New_script_Start
#     orr_log_final_objs_gaussian_splat_like(
#         objects, obj_classes,
#         samples_per_gaussian=80,
#         n_sigma_clip=2.5,
#         point_radius=0.01
#     )
# # New_script_End 

# New_script_Start
    orr_log_final_objs_gaussian_saved_samples(objects, obj_classes) # 이거는 단지 시각화
# New_script_End

# New_script_Start
    # obj["gaussians"]가 이미 채워진 상태에서
    # ply_path = str(exp_out_path / "final_gaussians" / "objects_gaussians.ply")
    out_dir = str(exp_out_path / "final_gaussians")
    # export_objects_gaussians_to_3dgs_ply(objects, ply_path, scale_multiplier=1.0, opacity_default=1.0)
    export_objects_gaussians_and_samples(objects, out_dir)
    
# New_script_End
 
    # # New_script_Start
    # orr_log_final_objs_gaussians(
    #     objects=objects,
    #     obj_classes=obj_classes,
    #     n_sigma=2.0,              # 2~3 추천 (2σ 박스)
    #     use_gaussian_rgb=True,    # 클러스터 평균색 사용
    #     log_object_label_box=False
    # )
    # # New_script_End
   
    
    # Consolidate captions 
    for object in objects:
        obj_captions = object['captions'][:20]
        # consolidated_caption = consolidate_captions(openai_client, obj_captions)
        consolidated_caption = []
        object['consolidated_caption'] = consolidated_caption

    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # # Save the pointcloud
    # if cfg.save_pcd:
    #     save_pointcloud(
    #         exp_suffix=cfg.exp_suffix,
    #         exp_out_path=exp_out_path,
    #         cfg=cfg,
    #         objects=objects,
    #         obj_classes=obj_classes,
    #         latest_pcd_filepath=cfg.latest_pcd_filepath,
    #         create_symlink=True,
    #         edges=map_edges
    #     )

    if cfg.save_json:
        save_obj_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects
        )
        
        save_edge_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects,
            edges=map_edges
        )

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()

if __name__ == "__main__":
    main()
