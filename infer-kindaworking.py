import torch
import argparse
import yaml
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
from PIL import Image
from matplotlib import cm
import numpy as np

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
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        
        print(type(image))
        # image = T.Resize((nH, nW))(image)
        image = cv2.resize(image.numpy(), (nH, nW), interpolation = cv2.INTER_AREA)
        console.print(f"resized")
        image = torch.tensor(image)
        print(image.shape)
        image = image.reshape(3, 512, 218464)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        orig_img = orig_img.squeeze()
        print(orig_img.shape)
        # print(seg_image.size())
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = draw_text(seg_image, seg_map, self.labels)
        return image

    # @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map
    
    def predict_camera(self, frame, overlay: bool) -> Tensor:
        # print("in")
        # # img = self.preprocess(torch.tensor(frame))
        # print("in2")
        seg_map = self.model_forward(frame)
        seg_map = self.postprocess(frame, seg_map, overlay)
        return seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    parser.add_argument('--camera', type=bool, default=False)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    semseg = SemSeg(cfg)

    if args.camera:
        cap = cv2.VideoCapture(0)
        print("WEBCAM STARTED ... ... ...")
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        count = 0 
        while(True):
            _, frame = cap.read()
            src = frame.copy()
        
            H, W = src.shape[:-1]
            console.print(f"Original Image Size > [red]{H}x{W}[/red]")
            # scale the short side of image to target size
            size = cfg['TEST']['IMAGE_SIZE']
            scale_factor = size[0] / min(H, W)
            nH, nW = round(H*scale_factor), round(W*scale_factor)
            # make it divisible by model stride
            nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
            console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
            # resize the image
            image = torch.tensor(src)
            image = image.reshape(3, 720, 1280)
            # # divide by 255, norm and add batch dim
            tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # T.Lambda(lambda x: x.unsqueeze(0))
            ])
            image = image.unsqueeze(0)
            # return image
            
            segmap = semseg.predict_camera(image.float(), cfg['TEST']['OVERLAY'])
            cv2.imshow('Input', np.array(segmap))
            c = cv2.waitKey(1)
            if c == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
            
    else:
        test_file = Path(cfg['TEST']['FILE'])
        print(test_file)
        if not test_file.exists():
            raise FileNotFoundError(test_file)
        save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
        save_dir.mkdir(exist_ok=True)
        
        semseg = SemSeg(cfg)
        with console.status("[bright_green]Processing..."):
            if test_file.is_file():
                console.rule(f'[green]{test_file}')
                segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
                segmap.save(save_dir / f"{str(test_file.stem)}.png")
            else:
                files = test_file.glob('*.*')
                for file in files:
                    console.rule(f'[green]{file}')
                    segmap = semseg.predict(str(file), cfg['TEST']['OVERLAY'])
                    segmap.save(save_dir / f"{str(file.stem)}.png")

    
        console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")