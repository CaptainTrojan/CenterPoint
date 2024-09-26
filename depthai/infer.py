import numpy as np
import onnxruntime as ort
import argparse
# import torch
import itertools
from visualization import visual
import os
from tqdm import tqdm
import shutil
import copy
from time import perf_counter
from collections import defaultdict
import pickle

# Constants
MAX_PILLARS = 30000
MAX_POINT_IN_PILLARS = 20
FEATURE_NUM = 10
BEV_W = 512
BEV_H = 512
X_MIN, X_MAX, X_STEP = -51.2, 51.2, 0.2
Y_MIN, Y_MAX, Y_STEP = -51.2, 51.2, 0.2
Z_MIN, Z_MAX = -5.0, 3.0
PI = 3.141592653

# Parameters for postprocess
SCORE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.2
INPUT_NMS_MAX_SIZE = 1000
OUT_SIZE_FACTOR = 4.0
TASK_NUM = 6
REG_CHANNEL = 2
HEIGHT_CHANNEL = 1
ROT_CHANNEL = 2
VEL_CHANNEL = 2
DIM_CHANNEL = 3
OUTPUT_H = 128
OUTPUT_W = 128

# Model outputs
REG_NAMES = ["594", "618", "642", "666", "690", "714"]
HEIGHT_NAMES = ["598", "622", "646", "670", "694", "718"]
ROT_NAMES = ["606", "630", "654", "678", "702", "726"]
VEL_NAMES = ["610", "634", "658", "682", "706", "730"]
DIM_NAMES = ["736", "740", "744", "748", "752", "756"]
SCORE_NAMES = ["737", "741", "745", "749", "753", "757"]
CLASS_NAMES = ["738", "742", "746", "750", "754", "758"]
CLASS_OFFSET_PER_TASK = [0, 1, 3, 5, 6, 8]


class Box:
    def __init__(self, x, y, z, l, h, w, theta, velX, velY, score, cls):
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.h = h
        self.w = w
        self.theta = theta
        self.velX = velX
        self.velY = velY
        self.score = score
        self.cls = cls
        self.isDrop = False

    def __repr__(self):
        return f"Box(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, l={self.l:.2f}, h={self.h:.2f}, "
        f"w={self.w:.2f}, theta={self.theta:.2f}, velX={self.velX:.2f}, velY={self.velY:.2f}, "
        f"score={self.score:.2f}, cls={self.cls:.2f}, isDrop={self.isDrop})"

    def __str__(self):
        return f"Box(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, l={self.l:.2f}, h={self.h:.2f}, "
        f"w={self.w:.2f}, theta={self.theta:.2f}, velX={self.velX:.2f}, velY={self.velY:.2f}, "
        f"score={self.score:.2f}, cls={self.cls:.2f}, isDrop={self.isDrop})"


def read_bin_file(filename):
    """
    Reads a binary file containing a point cloud.

    Args:
    - filename (str): Path to the binary file.

    Returns:
    - np.ndarray: Numpy array containing the point cloud data (N x 5).
    """
    with open(filename, 'rb') as f:
        file_content = f.read()

    # Each point has 5 features (assuming x, y, z, intensity, time or reflectance)
    feature_num = 5

    # Read the file content into a numpy array of floats
    point_cloud = np.frombuffer(file_content, dtype=np.float32).reshape(-1, feature_num)

    point_num = point_cloud.shape[0]
    print(f"[INFO] pointNum: {point_num}")

    return point_cloud


def rotate_around_center(box, corner, cos_val, sin_val):
    new_corners = np.zeros_like(corner)
    for idx in range(4):
        x, y = corner[idx]
        new_corners[idx][0] = (x - box.x) * cos_val + (y - box.y) * (-sin_val) + box.x
        new_corners[idx][1] = (x - box.x) * sin_val + (y - box.y) * cos_val + box.y
    return new_corners


def find_max_min(box, xy_idx):
    max_val = box[0][xy_idx]
    min_val = box[0][xy_idx]
    for idx in range(4):
        max_val = max(max_val, box[idx][xy_idx])
        min_val = min(min_val, box[idx][xy_idx])
    return max_val, min_val


def align_box(corner_rot):
    max_x, min_x = find_max_min(corner_rot, 0)
    max_y, min_y = find_max_min(corner_rot, 1)
    return np.array([[min_x, min_y], [max_x, max_y]])


def iou_bev(box_a, box_b):
    ax1, ax2 = box_a.x - box_a.l / 2, box_a.x + box_a.l / 2
    ay1, ay2 = box_a.y - box_a.w / 2, box_a.y + box_a.w / 2
    bx1, bx2 = box_b.x - box_b.l / 2, box_b.x + box_b.l / 2
    by1, by2 = box_b.y - box_b.w / 2, box_b.y + box_b.w / 2

    corner_a = np.array([[ax1, ay1], [ax1, ay2], [ax2, ay1], [ax2, ay2]])
    corner_b = np.array([[bx1, by1], [bx1, by2], [bx2, by1], [bx2, by2]])

    cos_a, sin_a = np.cos(box_a.theta), np.sin(box_a.theta)
    cos_b, sin_b = np.cos(box_b.theta), np.sin(box_b.theta)

    corner_a_rot = rotate_around_center(box_a, corner_a, cos_a, sin_a)
    corner_b_rot = rotate_around_center(box_b, corner_b, cos_b, sin_b)

    corner_align_a = align_box(corner_a_rot)
    corner_align_b = align_box(corner_b_rot)

    s_box_a = (corner_align_a[1][0] - corner_align_a[0][0]) * (corner_align_a[1][1] - corner_align_a[0][1])
    s_box_b = (corner_align_b[1][0] - corner_align_b[0][0]) * (corner_align_b[1][1] - corner_align_b[0][1])

    inter_w = np.min([corner_align_a[1][0], corner_align_b[1][0]]) - np.max([corner_align_a[0][0], corner_align_b[0][0]])
    inter_h = np.min([corner_align_a[1][1], corner_align_b[1][1]]) - np.max([corner_align_a[0][1], corner_align_b[0][1]])

    s_inter = np.maximum(inter_w, 0.0) * np.maximum(inter_h, 0.0)
    s_union = s_box_a + s_box_b - s_inter

    return s_inter / s_union


def aligned_nms_bev(pred_boxes, nms_threshold):
    if not pred_boxes:
        return

    pred_boxes.sort(key=lambda box: box.score, reverse=True)
    box_size = min(len(pred_boxes), INPUT_NMS_MAX_SIZE)

    for box_idx1 in range(box_size):
        for box_idx2 in range(box_idx1 + 1, box_size):
            if pred_boxes[box_idx2].isDrop:
                continue
            if iou_bev(pred_boxes[box_idx1], pred_boxes[box_idx2]) > nms_threshold:
                pred_boxes[box_idx2].isDrop = True


def convert_box(info):
    boxes = info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # dummy value
    detection['label_preds'] = np.zeros(len(boxes))
    detection['scores'] = np.ones(len(boxes))

    return detection


def read_boxes_from_txt(path):
    """
    Reads boxes from a .txt file and converts them into a list of Box objects.
    """
    boxes = []
    with open(path) as f:
        trt_res = f.readlines()

    for line in trt_res:
        box_data = [float(it) for it in line.strip().split(" ")[:9]]
        score = float(line.strip().split(" ")[-2])
        klass = int(line.strip().split(" ")[-1])

        # Convert to Box objects
        box = Box(x=box_data[0], y=box_data[1], z=box_data[2],
                  l=box_data[3], h=box_data[4], w=box_data[5],
                  theta=box_data[6], velX=box_data[7], velY=box_data[8],
                  score=score, cls=klass)
        boxes.append(box)

    return boxes


def process_boxes_to_dict(boxes, token):
    """
    Processes an existing list of Box objects and converts them into dictionary format
    compatible with the original `trt_pred` format.
    """
    box3d = []
    scores = []
    klass = []

    for box in boxes:
        box3d.append([box.x, box.y, box.z, box.l, box.h, box.w, box.theta, box.velX, box.velY])
        scores.append(box.score)
        klass.append(box.cls)

    trt_pred = {
        # "box3d_lidar": torch.from_numpy(np.array(box3d)),
        # "scores": torch.from_numpy(np.array(scores)),
        # "label_preds": torch.from_numpy(np.array(klass, np.int32)),
        "box3d_lidar": np.array(box3d),
        "scores": np.array(scores),
        "label_preds": np.array(klass, np.int32),
        "metadata": {
            "num_point_features": 5,
            "token": token
        }
    }

    return trt_pred


def read_txt_result(path):
    """
    Reads a .txt file and processes it into the original `trt_pred` format.
    """
    token = path.split("/")[-1].split(".")[0]
    boxes = read_boxes_from_txt(path)
    trt_pred = process_boxes_to_dict(boxes, token)
    return trt_pred, token


def build_nuscenes_dataset(args):
    from det3d.datasets import build_dataset
    from det3d.torchie import Config

    data_root = args.data
    dataset_type = "NuScenesDataset"
    val_anno = f"{data_root}/infos_val_10sweeps_withvelo_filter_True.pkl"
    nsweeps = 10
    tasks = [
        dict(num_class=1, class_names=["car"]),
        dict(num_class=2, class_names=["truck", "construction_vehicle"]),
        dict(num_class=2, class_names=["bus", "trailer"]),
        dict(num_class=1, class_names=["barrier"]),
        dict(num_class=2, class_names=["motorcycle", "bicycle"]),
        dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    ]

    class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
    val_preprocessor = dict(
        mode="val",
        shuffle_points=False,
    )

    voxel_generator = dict(
        range=[X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX],
        voxel_size=[0.2, 0.2, 8],
        max_points_in_voxel=20,
        max_voxel_num=[30000, 60000],
    )
    target_assigner = dict(
        tasks=tasks,
    )
    assigner = dict(
        target_assigner=target_assigner,
        out_size_factor=4,  # For PointPillars
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
    )
    test_pipeline = [
        dict(type="LoadPointCloudFromFile", dataset=dataset_type),
        dict(type="LoadPointCloudAnnotations", with_bbox=True),
        dict(type="Preprocess", cfg=val_preprocessor),
        dict(type="Voxelization", cfg=voxel_generator),
        dict(type="AssignLabel", cfg=assigner),
        dict(type="Reformat"),
    ]

    cfg_data_val = dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        version="v1.0-mini"
    )

    cfg = Config(cfg_dict={'val': cfg_data_val})
    dataset = build_dataset(cfg.val)

    return dataset


def build_data_pipeline(args):
    # cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini.py')
    # dataset = build_dataset(cfg.data.val)

    if args.simple_data:
        with open(args.data, 'rb') as f:
            data = pickle.load(f)
        return len(data), data
    else:
        from det3d.torchie.parallel import collate_kitti
        from torch.utils.data import DataLoader

        dataset = build_nuscenes_dataset(args)

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=None,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_kitti,
            pin_memory=False,
        )

        return len(dataset), nuscenes_iterator(dataset, data_loader)


def nuscenes_iterator(dataset, data_loader):
    for i, data_batch in enumerate(data_loader):
        token = data_batch['metadata'][0]['token']
        points = data_batch['points'][:, 1:].cpu().numpy()
        info = dataset._nusc_infos[i]
        yield token, points, info


def format_detections_for_evaluation(args, detections):
    if detections is None:
        detections = {}
        for fn in os.listdir(args.result_dir):
            path = os.path.join(args.result_dir, fn)
            output, token = read_txt_result(path)
            detections[token] = output
    else:
        for token, output in detections.items():
            detections[token] = process_boxes_to_dict(output, token)
    return detections


class InferencePipeline:
    def run_inference(self, features, indices):
        """
        Run inference using the model.

        Args:
            features (numpy.ndarray): Input features for the inference.
            indices (numpy.ndarray): Input indices for the inference.

        Returns:
            dict: Dictionary containing the raw output of the inference, mapped to their respective names ('594' -> np.ndarray, etc.).
        """
        raise NotImplementedError("Pipeline must implement run_inference method")

    def preprocess(self, points: np.ndarray):
        """
        Preprocesses the input points by performing the following steps:
        1. Filters out points that are outside the bounding box.
        2. Calculates the indices in the BEV (Bird's Eye View) grid for the valid points.
        3. Ensures that the indices are within the grid limits.
        4. Calculates unique pillar positions.
        5. Stores points in pillars.
        6. Computes mean features for each pillar and normalizes them.

        Args:
            points (np.ndarray): Input points as a numpy array of shape (N, 3), where N is the number of points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the preprocessed features and indices.
            - features (np.ndarray): Preprocessed features as a numpy array of shape (FEATURE_NUM, MAX_PILLARS, MAX_POINT_IN_PILLARS).
            - indices (np.ndarray): Preprocessed indices as a numpy array of shape (MAX_PILLARS * 2,).
        """
        # Initialize features and indices arrays
        features = np.zeros((FEATURE_NUM, MAX_PILLARS, MAX_POINT_IN_PILLARS), dtype=np.float32)
        indices = np.full((MAX_PILLARS * 2,), -1, dtype=np.int64)

        # Initialize point count array
        point_count = np.zeros((MAX_PILLARS,), dtype=np.int32)

        # Extract x, y, z coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Check if points are within the bounding box
        mask = (X_MIN <= x) & (x <= X_MAX) & (Y_MIN <= y) & (y <= Y_MAX) & (Z_MIN <= z) & (z <= Z_MAX)
        valid_points = points[mask]

        # Calculate indices in the BEV grid for valid points
        x_valid = valid_points[:, 0]
        y_valid = valid_points[:, 1]
        x_idx = ((x_valid - X_MIN) / X_STEP).astype(np.int32)
        y_idx = ((y_valid - Y_MIN) / Y_STEP).astype(np.int32)

        # Ensure indices are within grid limits
        valid_mask = (0 <= x_idx) & (x_idx < BEV_W) & (0 <= y_idx) & (y_idx < BEV_H)
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        valid_points = valid_points[valid_mask]

        # Calculate unique pillar positions
        position_id = y_idx * BEV_W + x_idx

        # Create a mapping from position_id to pillar_id
        pillar_unique_id = np.full((BEV_H * BEV_W,), -1, dtype=np.int32)  # Initialize with -1
        current_pillar_count = 0

        # Store points in pillars
        for idx in range(position_id.size):
            pos_id = position_id[idx]

            # If the position_id is not yet in the pillar_unique_id
            if pillar_unique_id[pos_id] == -1:
                if current_pillar_count >= MAX_PILLARS:
                    break  # End if we reach the maximum number of pillars
                pillar_unique_id[pos_id] = current_pillar_count
                indices[current_pillar_count * 2 + 1] = pos_id  # Set the pillar index in indices
                current_pillar_count += 1

            # Retrieve the current pillar index
            pillar_id = pillar_unique_id[pos_id]
            n_points = point_count[pillar_id]  # Use point_count to get the number of points in this pillar

            if n_points < MAX_POINT_IN_PILLARS:
                # Store features in the appropriate pillar
                features[:5, pillar_id, n_points] = valid_points[idx]
                point_count[pillar_id] += 1  # Increment the point count for this pillar

        # Compute mean features for each pillar and normalize
        for pillar_id in range(current_pillar_count):
            n_points = point_count[pillar_id]  # Get the count of points in this pillar
            if n_points > 0:
                # Extract features for the current pillar
                pillar_points = features[:, pillar_id, :n_points]
                pillar_mean = pillar_points.mean(axis=1)

                # Normalize features
                features[5:8, pillar_id, :n_points] = pillar_points[:3] - pillar_mean[:3][:, None]

        return features, indices

    def postprocess(self, raw_output):
        pred_result = []

        for task_idx in range(TASK_NUM):
            pred_boxes = []

            # Accessing buffers directly from the raw_output dictionary and removing the batch dimension
            reg = raw_output[REG_NAMES[task_idx]][0]  # shape: (2, 128, 128)
            height = raw_output[HEIGHT_NAMES[task_idx]][0]  # shape: (1, 128, 128)
            rot = raw_output[ROT_NAMES[task_idx]][0]  # shape: (2, 128, 128)
            vel = raw_output[VEL_NAMES[task_idx]][0]  # shape: (2, 128, 128)
            dim = raw_output[DIM_NAMES[task_idx]][0]  # shape: (3, 128, 128)
            score = raw_output[SCORE_NAMES[task_idx]][0]  # shape: (128, 128)
            klass = raw_output[CLASS_NAMES[task_idx]][0]  # shape: (128, 128)

            # Use numpy's multi-dimensional indexing to find valid indices
            valid_indices = np.argwhere(score >= SCORE_THRESHOLD)  # shape: (N, 2) with valid (y_idx, x_idx)

            for y_idx, x_idx in valid_indices:
                # Compute positions and dimensions using fancy indexing
                x = (x_idx + reg[0, y_idx, x_idx]) * OUT_SIZE_FACTOR * X_STEP + X_MIN
                y = (y_idx + reg[1, y_idx, x_idx]) * OUT_SIZE_FACTOR * Y_STEP + Y_MIN
                z = height[0, y_idx, x_idx]

                if not (X_MIN < x < X_MAX and Y_MIN < y < Y_MAX and Z_MIN < z < Z_MAX):
                    continue

                box = Box(
                    x=x, y=y, z=z,
                    l=dim[0, y_idx, x_idx],
                    h=dim[1, y_idx, x_idx],
                    w=dim[2, y_idx, x_idx],
                    theta=np.arctan2(rot[0, y_idx, x_idx], rot[1, y_idx, x_idx]),
                    velX=vel[0, y_idx, x_idx],
                    velY=vel[1, y_idx, x_idx],
                    score=score[y_idx, x_idx],
                    cls=klass[y_idx, x_idx] + CLASS_OFFSET_PER_TASK[task_idx]
                )

                pred_boxes.append(box)

            aligned_nms_bev(pred_boxes, NMS_THRESHOLD)

            for box in pred_boxes:
                if not box.isDrop:
                    pred_result.append(box)

        return pred_result

    def infer_from_point_cloud(self, point_cloud):
        """
        Complete inference pipeline from point cloud data.
        """
        times = {}

        start = perf_counter()
        features, indices = self.preprocess(point_cloud)
        times['preprocess'] = perf_counter() - start

        start = perf_counter()
        raw_output = self.run_inference(features, indices)
        times['inference'] = perf_counter() - start

        start = perf_counter()
        result = self.postprocess(raw_output)
        times['postprocess'] = perf_counter() - start

        return result, times


class ONNXInferencePipeline(InferencePipeline):
    def __init__(self, model_path):
        # Load the ONNX model using onnxruntime
        self.session = ort.InferenceSession(model_path)

    def run_inference(self, features, indices):
        # Run inference with the ONNX model
        features_input = self.session.get_inputs()[0]
        indices_input = self.session.get_inputs()[1]
        features_reshaped = features.reshape(features_input.shape)
        indices_reshaped = indices.reshape(indices_input.shape)
        raw_output = self.session.run(None, {features_input.name: features_reshaped, indices_input.name: indices_reshaped})

        # Map outputs to names
        output_names = [output.name for output in self.session.get_outputs()]
        raw_output = {name: output for name, output in zip(output_names, raw_output)}

        return raw_output


def run_inference(args):
    # Initialize the inference pipeline
    inference_pipeline = ONNXInferencePipeline(args.model)

    # Initialize the data pipeline
    n_elems, iterable = build_data_pipeline(args)

    limit = n_elems if args.max_elems is None else min(args.max_elems, n_elems)
    detections = {}
    avg_times = defaultdict(float)

    for i, (token, points, info) in tqdm(enumerate(iterable), desc="Running inference", total=limit):
        if args.max_elems is not None and i >= args.max_elems:
            break
        output, times = inference_pipeline.infer_from_point_cloud(points)

        detections[token] = output
        for phase in times:
            avg_times[phase] = avg_times[phase] + times[phase]

    for phase in avg_times:
        avg_times[phase] /= len(detections)
        print(f"Average time for {phase}: {avg_times[phase]:.4f} seconds")

    return detections


def run_evaluation(args, output_dict):
    print(f"Running evaluation and visualization, outputs in {args.output_dir}")

    # Clean the output directory
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=False)

    # Format detections
    output_dict = format_detections_for_evaluation(args, output_dict)

    # Initialize variables
    points_dict = {}
    points_list = []
    token_list = []
    gt_annos_dict = {}

    n_elems, iterable = build_data_pipeline(args)
    for i, (token, points, info) in enumerate(iterable):
        token_list.append(token)
        points_dict[token] = points.T
        gt_annos_dict[token] = convert_box(info)

    detections = {}
    detections_for_draw = []
    gt_annos = []

    for token in token_list:
        assert token in output_dict, f"Token {token} not found in output_dict!. Ensure that you ran the inference on the same data as you're evaluating on."
        # if token not in output_dict:
        #     continue
        output = output_dict[token]

        points_list.append(points_dict[token])
        gt_annos.append(gt_annos_dict[token])
        for k, v in output.items():
            if k not in [
                "metadata",
            ]:
                output[k] = v
        detections_for_draw.append(output)
        detections.update(
            {token: output}
        )

    if not args.simple_data:
        from det3d.torchie.trainer.utils import all_gather

        all_predictions = all_gather(detections)

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        result_dict, _ = build_nuscenes_dataset(args).evaluation(copy.deepcopy(predictions), output_dir=args.output_dir, testset=False)

        if result_dict is not None:
            for k, v in result_dict["results"].items():
                print(f"Evaluation {k}: {v}")

    draw_num = len(points_list)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=False)
    for i in tqdm(range(draw_num), desc="Visualizing results"):
        visual(points_list[i], gt_annos[i], detections_for_draw[i], i, save_path=image_dir)


def export_detections(args, detections):
    print(f"Saving results to {args.result_dir}")

    # Clean the result directory
    shutil.rmtree(args.result_dir, ignore_errors=True)
    os.makedirs(args.result_dir, exist_ok=False)

    # Export
    for token, output in detections.items():
        with open(f"{args.result_dir}/{token}.txt", "w") as f:
            for box in output:
                f.write(f"{box.x:.6f} {box.y:.6f} {box.z:.6f} {box.l:.6f} {box.h:.6f} {box.w:.6f} "
                        f"{box.velX:.6f} {box.velY:.6f} {box.theta:.6f} {box.score:.6f} {box.cls}\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
        This program supports the following modes of operation:
        1. Run inference with the model and data, without saving or evaluating/visualizing the results.
        2. Run inference with the model and data, optionally save the results to result_dir.
        3. Run inference with the model and data, and optionally evaluate/visualize the results in output_dir.
        4. Run inference, save results to result_dir, and optionally evaluate/visualize the results in output_dir.
        5. Only evaluate/visualize existing results from result_dir, and output to output_dir.

        Usage Rules:
        - If you specify --model and --data, you can run inference.
        - If you specify --result_dir, results from inference can be saved.
        - If you specify --output_dir, results can be evaluated or visualized.
        - To only evaluate/visualize existing results, specify --data, --result_dir and --output_dir without the --model.
        """)

    # Primary args
    parser.add_argument("-m", "--model", help="Path to the model")
    parser.add_argument("-d", "--data", help="Path to the nuScenes data directory (or .pkl file in case of simple data pipeline)")
    parser.add_argument("-r", "--result_dir", help="Path to the result directory")
    parser.add_argument("-o", "--output_dir", help="Path to the output directory")
    # Secondary args
    parser.add_argument("--simple_data", action="store_true", help="Use simple data pipeline (binary files). Turns off "
                                                                   "evaluation (visualization is still possible).")
    parser.add_argument("--max_elems", type=int, help="Maximum number of elements to process. Turns off evaluation (visualization is still possible).")
    args = parser.parse_args()

    # Argument verification tree
    if args.model:
        if not args.data:
            print("Error: Model provided without data directory. Add --data to run inference.")
            exit(1)

        # Scenario 1: Run inference (model + data), no save, no eval/visualize
        if not args.result_dir and not args.output_dir:
            print("Running inference with the model and data, no saving or visualization.")

        # Scenario 2: Run inference and optionally save results
        elif args.result_dir and not args.output_dir:
            print(f"Running inference with the model and data, saving results to {args.result_dir}.")

        # Scenario 3: Run inference and optionally evaluate/visualize
        elif not args.result_dir and args.output_dir:
            print(f"Running inference with the model and data, visualizing results in {args.output_dir}.")

        # Scenario 4: Run inference, save results, and optionally evaluate/visualize
        elif args.result_dir and args.output_dir:
            print(f"Running inference with the model and data, saving results to {args.result_dir} and visualizing them in {args.output_dir}.")

        should_run_inference = True
    elif args.data and args.result_dir and args.output_dir:
        # Scenario 5: Only eval/visualize from existing results
        print(f"Evaluating and visualizing existing results from {args.result_dir} and saving outputs to {args.output_dir}.")
        should_run_inference = False

    else:
        # Invalid combination of parameters, print warning and quit
        print("Error: Invalid combination of parameters.")
        print("Ensure the following valid configurations:")
        print("  1. Provide --model and --data for inference.")
        print("  2. Optionally provide --result_dir to save inference results.")
        print("  3. Optionally provide --output_dir to evaluate/visualize inference results.")
        print("  4. If not running inference, provide --data, --result_dir and --output_dir for evaluation/visualization only.")
        exit(1)

    detections = None
    if should_run_inference:
        detections = run_inference(args)

        if args.result_dir:
            # Save the results to the result directory
            export_detections(args, detections)

    if args.output_dir:
        # Evaluate and visualize
        run_evaluation(args, detections)


if __name__ == '__main__':
    main()
