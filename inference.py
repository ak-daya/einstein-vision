from midas_model import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmdet_model import *
from dataloaders import *
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Init models
depth_model = MidasModel(device, "DPT_Large", depth_scale=10)
bb_model = MmdetModel(device)

labels = np.array(read_names_file("coco.names"))

for i in tqdm(range(1, 14)):

    scene = f"scene{i}"
    base_path = Path.cwd() / "scenes" / scene
        
    dir_to_images = base_path / "front" / "raw"
    path_to_bb = base_path / "front" / "boundingbox" / f"bb.json"
    dir_to_depth_imgs = base_path / "front" / "depth"

    frames = LoadImagesFromFolder(dir_to_images)
    
    data = []

    for j in range(len(frames)):
        json_dict = {"Frame" : j}
        frame = frames[j]

        # Depth estimation
        depth_img = depth_model.test_midas([frame])
        img_path = dir_to_depth_imgs / f'{j}.jpg'
        cv2.imwrite(str(img_path), depth_img)

        # Bounding boxes
        bounding_boxes, _ = bb_model.test_mmdet([frame])
        pred_scores = bounding_boxes[0]['predictions'][0]['scores']
        pred_labels = np.array(bounding_boxes[0]['predictions'][0]['labels'])
        pred_bound_boxes = bounding_boxes[0]['predictions'][0]['bboxes']

        json_dict["Scores"] = pred_scores
        json_dict["Labels"] = labels[pred_labels].tolist()
        json_dict["Boxes"] = pred_bound_boxes

        data.append(json_dict)

    with path_to_bb.open("w") as file:
        json.dump(data, file, indent=4)