import torch
import argparse
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import cv2

from rich.console import Console
console = Console()

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image):
        W, H, _ = image.shape
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = cv2.resize(image,(nH,nW), interpolation = cv2.INTER_AREA)
        image = torch.tensor(image)
        image = image.permute((2, 0, 1))
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)

        # get prediction probability of each class for each pixel
        prob_logit = seg_map[0].reshape((720,1280,150))
        prob_logit = prob_logit.softmax(dim=2)  # shape: (720, 1280, 150)

        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # get predicted class labels for each pixel
        labels_logit = seg_map[0] # shape: (720, 1280)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = draw_text(seg_image, seg_map, self.labels)
        return image, prob_logit, labels_logit

    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, image, overlay: bool):
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.permute((2, 0, 1))
        seg_map, prob_logit, labels_logit = self.postprocess(image_tensor, seg_map, overlay)
        return seg_map, prob_logit, labels_logit