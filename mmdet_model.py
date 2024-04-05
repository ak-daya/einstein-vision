from mmdet.apis import DetInferencer
import numpy as np

class MmdetModel:
    def __init__(self, device):
        self.device = device
        self.model = DetInferencer(model = 'rtmdet_tiny_8xb32-300e_coco', device = self.device)

    def test_mmdet(self, images):
        
        obj_det = []
        
        inferencer = DetInferencer(model = 'rtmdet_tiny_8xb32-300e_coco', device = 'cpu')
        for i,img in enumerate(images):
            result = inferencer(img, return_vis=True)
            result_img = np.array(result['visualization'])
            obj_det.append(result)
            
        return obj_det, result_img