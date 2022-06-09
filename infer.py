from tkinter.tix import IMAGETEXT
import torch
import argparse
import yaml
import math
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semsegmentation import SemSeg
import cv2
from PIL import Image
import numpy as np
import time

from rich.console import Console
console = Console()

def run(args):
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    cap = cv2.VideoCapture(0)
    print("WEBCAM STARTED ... ... ...")
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")
    
    semseg = SemSeg(cfg)

    while(True):
        _,frame = cap.read()
        image = frame.copy()
        segmap = semseg.predict(image=image, overlay=cfg['TEST']['OVERLAY'])
        cv2.imshow('Segmentation', np.array(segmap))
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()
    run(args)

    