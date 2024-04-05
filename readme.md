# Einstein Vision

## Pipeline
1. Create frames from video sequences (saved in .scenes/scene{i}/front/raw/) using video2img.py
2. Run inference on all scenes using inference.py which creates
    - bounding box json (saved in .scenes/scene{i}/front/boundingbox/) 
    - grayscale 8-bit depth images (saved in .scenes/scene{i}/front/depth/)
    - Note: This takes forever (~8 hours for 13 scenes), that's why inference has been separated from processing world points from inferred data
3. Estimate world points for each object for each frame on all scenes using scene_estimation.py. This uses the bounding box json and depth images from the previous part.
    - Creates a render.json file for each scene (saved in .scenes/scene{i}/) 
    - Note: Relatively faster process
4. Render a 3D scene for each frame on all scenes using Scene.blend/render.py script
    - Creates a rendered image for each frame (saved in .scenes/scene{i}/front/renders)
5. Render videos of the raw and rendered images, and one video comparing them side by side using Video.blend/video.py. This creates
    - Video of the raw images (saved in .scenes/scene{i}/raw.mp4)
    - Video of the raw images (saved in .scenes/scene{i}/render.mp4)
    - Video comparing both (saved in .scenes/scene{i}/comparison.mp4)


## File structure
1. blender
    - asset-jsons: mesh data of each blender object type as JSON
    - Scene.blend: main rendering environment with scripts included
    - test.blend: temp file

2. scenes
    - scene{x}:
        - front
            - boundingbox: contains bb.json file which contains mmdet output
            - depth: contains grayscale depth images normalized [0, 1]
            - raw: contains RGB frames
            - renders: contains blender renders (same size as frames)

3. calibration
    