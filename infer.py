from tkinter.tix import IMAGETEXT
import argparse
import yaml
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semsegmentation import SemSeg
import cv2
import numpy as np
# from realsense_camera import *
from fps import FPS
import time

from rich.console import Console
console = Console()

def run(args):
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    cap = cv2.VideoCapture(18)
    # cap.set(cv2.CAP_PROP_FPS, 2)
    print("WEBCAM STARTED ... ... ...")
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")
    
    semseg = SemSeg(cfg)

    while(True):
        # _,bgr_frame = cap.read()
        # ret, bgr_frame, depth_frame = rs.get_frame_stream()
        ret, bgr_frame = cap.read()
        time.sleep(1)
        image = bgr_frame.copy()
        total_fps.start()
        segmap = semseg.predict(image=image, overlay=cfg['TEST']['OVERLAY'])
        segmap = np.array(segmap)
        total_fps.stop()
    
        cv2.putText(segmap, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Segmentation', segmap)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()
    # Load Realsense camera
    total_fps = FPS()
    # rs = RealsenseCamera()
    run(args)

    