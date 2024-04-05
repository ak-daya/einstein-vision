import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

def normalize_linear(img, min=0, max=255):
    normalized = (img - img.min())/(img.max() - img.min()) * (max-min) +  min
    return normalized

def normalize_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(img)
    return normalized

def normalize_equHist(img):
    normalized = cv2.equalizeHist(img)
    return normalized

class MidasModel:
    def __init__(self, device, modelType, depth_scale):
        #creating midas model
        self.device = device
        self.modelType = modelType
        #keeping it in eval as we're not training
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True).to(self.device)
        self.model.eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        self.depth_scale = depth_scale
        
    def test_midas(self, images):

        if self.modelType in ["DPT_Large", "DPT_Hybrid"]:
            transform = self.transforms.dpt_transform
        else:
            transform = self.transforms.small_transform
        
        #loading Images
        depth = []
        
        #sending images through the model
        for img in images:
            input_batch = transform(img).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            output = prediction.cpu().numpy()   # inverse depth
            
            # data processing
            output = normalize_linear(output, min=0., max=255.)
            # invert to get depth
            output = output.max() - output      
            depth.append(output)
        
        depth = np.squeeze(np.array(depth))
            
        return depth