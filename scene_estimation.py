import numpy as np
from dataloaders import *
import json
from pathlib import Path
from tqdm import tqdm

class Camera:
    def __init__(self, name, rotation_in_world, location_in_world, intrinsics_matrix):
        self.name = name
        self.rotation = rotation_in_world
        self.location = location_in_world
        self.K = intrinsics_matrix        

def normalize_linear(img, min=0, max=255):
    normalized = (img - img.min())/(img.max() - img.min()) * (max-min) +  min
    return normalized

def sigmoid(x, x_scale=1., x_shift=0.):
    return 1/ (1 + np.exp(-(x_scale*(x - x_shift))))

def estimate_object_pose(camera : Camera, pixel_coords, depth):
    # Get the pixel coordinates
    u, v = pixel_coords

    # Convert image to camera projection (x, y, 1)
    x_projection = np.linalg.inv(camera.K) @ np.array([u, v, 1]).T

    # Get point in camera coordinates
    X_cam = depth * x_projection

    # Get point in world coordinates
    X_world = camera.location + camera.rotation @ (X_cam)

    return X_world

def main():

    front_cam = Camera(name="front", 
                       rotation_in_world=np.array([[1.,0.,0.,], [0.,0.,1], [0.,-1.,0.]]),
                       location_in_world=np.array([0., 0.24, 1.43]),
                       intrinsics_matrix=np.array(
                           [[1594.7, 0.00000000e+00, 655.3],
                           [0.00000000e+00, 1607.7, 414.4],
                           [0.00000000e+00, 0.00000000e+00, 1.0]]
                        )
                )
    
    labels_desired = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'traffic light', 'stop sign']
    labels_vehicles = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

    # object heights in meters
    heights = {
        'car' : 1.43,
        'truck' : 4.12,
        'bus' : 4.12,
        'motorcycle': 1.1,
        'bicycle' : 1.05,
        'person' : 1.75,
        'traffic light' : 0.762,
        'stop sign' : 2.13
    }

    # tuning parameters
    p_threshold = 0.54
    bb_depth_scale = 2250.
    midas_depth_scale = 50.

    # Loop over 13 scenes
    for i in tqdm(range(1, 13)):
        # path of bounding box and depth data
        scene_name = f"scene{i}"
        base_path = Path.cwd() / "scenes" / scene_name
        path_to_bb_data = base_path / "front" / "boundingbox" / f"bb.json"
        dir_to_depth_imgs = base_path / "front" / "depth"

        # path of output (render json)
        path_to_render_json = base_path / f"render.json"
        
        # Init render_json file for this scene which contains object data for each frame
        render_json = []
        
        # Load bounding box and depth data for all frames in scene
        with path_to_bb_data.open('r') as file:
            bb_data = json.load(file)
        depth_imgs = LoadImagesFromFolder(dir_to_depth_imgs, grayscale=True)

        # Loop over all frames
        for j in range(len(depth_imgs)):
            depth_img = depth_imgs[j]
            depth_img = normalize_linear(depth_img, min=0., max=1.)
            bb = bb_data[j]

            objects = []

            # Get bounding box data
            scores = bb['Scores']
            labels = bb['Labels']
            bounding_boxes = bb['Boxes']

            # Loop over all objects in frame
            for index, score in enumerate(scores):
                label = labels[index]
                if label not in labels_desired:
                    continue
                
                if score < p_threshold:
                    continue

                # Get bounding box boundary values
                box = bounding_boxes[index]                     # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                u, v = (x1+x2)/2, (y1+y2)/2     # box center for depth
                u_base, v_base = (x1+x2)/2, y2      # box bottom for position

                # Get world depth - Z
                h = y2 - y1
                
                # sigmoidal function for biasing bounding box or midas depth estimation
                # big h -> 1 : bounding box
                # small h -> 0 : midas
                closeness_bias = sigmoid(h, x_scale=0.05, x_shift=130)
                if label == "stop sign":
                    closeness_bias = 1

                midas_depth = depth_img[int(v)][int(u)]
                bb_depth_ratio = heights[label]/h
                depth = (1-closeness_bias) *  bb_depth_scale * bb_depth_ratio + closeness_bias * midas_depth_scale * midas_depth

                point = estimate_object_pose(front_cam, (u_base,v_base), depth)
                
                # Constrain vehicles to stay parallel to the ground
                if label in labels_vehicles:
                    point[-1] = 0.

                point = point.tolist()

                R = np.eye(3)
                R = R.tolist()

                objects.append({"Class": label, "Location": point, "R": R})

            # Populate render JSON file with frame and object data
            frame_data = {"Frame" : j}            
            frame_data["Objects"] = objects
            render_json.append(frame_data)

        with path_to_render_json.open("w") as file:
            json.dump(render_json, file, indent=4)

if __name__=="__main__":
    main()