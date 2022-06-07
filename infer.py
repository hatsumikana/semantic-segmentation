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
from semsegmentation import SemSeg

from rich.console import Console
console = Console()

def run(args):
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()
    run(args)

    