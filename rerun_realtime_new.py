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

# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter

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
    orr_log_vlm_image
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
from conceptgraph.dataset.datasets_common import get_dataset
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

def export_objects_gaussians_to_3dgs_ply(
    objects,
    out_ply_path: str,
    scale_multiplier: float = 1.0,
    opacity_default: float = 1.0,
):
    """
    Export all obj["gaussians"] into a single .ply in a format commonly used by 3DGS viewers.
    We will use SH degree 0 only (constant color) to keep it simple.

    Properties written:
      x y z
      nx ny nz              (dummy 0)
      f_dc_0 f_dc_1 f_dc_2   (RGB as DC term)
      opacity
      scale_0 scale_1 scale_2
      rot_0 rot_1 rot_2 rot_3   (quat xyzw)

    Note: Some viewers expect 'scale' in log-space. If your viewer shows weird sizes,
          we can switch to log(scale). First try linear scale.
    """
    out_ply_path = Path(out_ply_path)
    out_ply_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for obj in objects:
        if obj.get("num_detections", 1) < 1:
            continue
        if obj.get("is_background", False):
            continue
        gaussians = obj.get("gaussians", None)
        if not gaussians:
            continue

        for g in gaussians:
            mu = np.asarray(g["mu"], dtype=np.float32)
            Sigma = np.asarray(g["Sigma"], dtype=np.float32)

            R, scale = _sigma_to_rot_scale(Sigma)
            scale = scale * scale_multiplier
            quat = _rotmat_to_quat_xyzw(R)

            rgb01 = np.asarray(g.get("rgb", [1,1,1]), dtype=np.float32)
            rgb01 = np.clip(rgb01, 0.0, 1.0)

            # SH degree 0 only: DC term. Many 3DGS codes store SH in a specific basis.
            # For a quick visualization, most viewers will still show something reasonable.
            f_dc = rgb01  # (3,)

            opacity = float(g.get("alpha", opacity_default))

            rows.append([
                mu[0], mu[1], mu[2],
                0.0, 0.0, 0.0,               # nx ny nz dummy
                f_dc[0], f_dc[1], f_dc[2],    # f_dc_0..2
                opacity,
                scale[0], scale[1], scale[2],
                quat[0], quat[1], quat[2], quat[3],
            ])

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

    with open(out_ply_path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(map(lambda x: f"{x:.6f}", r.tolist())) + "\n")

    print(f"[INFO] Exported {rows.shape[0]} gaussians to: {out_ply_path}")
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

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

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

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
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

    # # New_script_Start
    # for obj in objects:
    #     pcd = obj.get("pcd", None)
    #     if pcd is None or len(pcd.points) < 4:
    #         continue  # 점 너무 적으면 skip

    #     # ✅ OBB 생성 (pcd 기반)
    #     obb = pcd.get_oriented_bounding_box()

    #     # (선택) 색상 맞추기
    #     if len(pcd.colors) > 0:
    #         obb.color = np.asarray(pcd.colors)[0]

    #     # ✅ bbox(AABB)는 건드리지 않고, OBB는 다른 키에 저장
    #     obj["obb"] = obb

    #     # ✅ JSON 저장까지 고려하면 open3d 객체 말고 파라미터도 같이 저장(강추)
    #     obj["obb_center"] = obb.center.tolist()
    #     obj["obb_extent"] = obb.extent.tolist()
    #     obj["obb_R"] = obb.R.tolist()
    
    # orr_log_final_objs_pcd_and_obb(objects, obj_classes)
    # # New_script_End

    
    # New_script_Start
    # =========================
    # Coarse Gaussian Mixture (Object-level) from obj["pcd"]
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
    ):
        """
        Returns: gaussians(list[dict]), used_points(int)
        Each gaussian dict: {mu(3), Sigma(3x3), rgb(3), w, alpha}
        rgb in 0~1
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

        if K == 1:
            mu = pts_fit.mean(axis=0)
            Xm = pts_fit - mu
            Sigma = (Xm.T @ Xm) / max(len(pts_fit) - 1, 1)
            Sigma = Sigma + np.eye(3) * reg_covar
            rgb = cols_fit.mean(axis=0) if cols_fit is not None else np.array([1.0, 1.0, 1.0])
            return [{
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "rgb": rgb.tolist(),
                "w": 1.0,
                "alpha": 1.0,
            }], Nf

        # 4) cluster points -> labels
        if _USE_SKLEARN:
            km = KMeans(n_clusters=K, n_init="auto", random_state=seed)
            labels = km.fit_predict(pts_fit)
        else:
            labels = _torch_kmeans(pts_fit, K=K, iters=30, seed=seed)

        # 5) compute Gaussian per cluster
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

            rgb = cols_fit[idx].mean(axis=0) if cols_fit is not None else np.array([1.0, 1.0, 1.0])
            w = float(idx.size / Nf)

            gaussians.append({
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "rgb": rgb.tolist(),
                "w": w,
                "alpha": 1.0,
            })

        # Edge-case: if all clusters got pruned
        if len(gaussians) == 0:
            mu = pts_fit.mean(axis=0)
            Xm = pts_fit - mu
            Sigma = (Xm.T @ Xm) / max(len(pts_fit) - 1, 1)
            Sigma = Sigma + np.eye(3) * reg_covar
            rgb = cols_fit.mean(axis=0) if cols_fit is not None else np.array([1.0, 1.0, 1.0])
            gaussians = [{
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "rgb": rgb.tolist(),
                "w": 1.0,
                "alpha": 1.0,
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

    for obj in objects:
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
        )

        obj["gaussians"] = gaussians
        obj["gaussian_meta"] = {
            "used_points": int(usedN),
            "voxel_size": float(_GM_VOXEL),
            "k_min": int(_GM_KMIN),
            "k_max": int(_GM_KMAX),
            "pts_per_gaussian": int(_GM_PTS_PER),
        }

    print("[INFO] Built coarse Gaussian mixtures for objects (obj['gaussians']).")
# New_script_End

# New_script_Start
    orr_log_final_objs_gaussian_splat_like(
        objects, obj_classes,
        samples_per_gaussian=80,
        n_sigma_clip=2.5,
        point_radius=0.01
    )
# New_script_End 

# New_script_Start
    # obj["gaussians"]가 이미 채워진 상태에서
    ply_path = str(exp_out_path / "final_gaussians" / "objects_gaussians.ply")
    export_objects_gaussians_to_3dgs_ply(objects, ply_path, scale_multiplier=1.0, opacity_default=1.0)
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
