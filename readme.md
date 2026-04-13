# Fast-SCNN (PyTorch)

PyTorch implementation of **Fast-SCNN** for fast semantic segmentation, based on the paper below.

## Paper

[Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/pdf/1902.04502.pdf)

## Repository layout

| Path | Description |
|------|-------------|
| `models/` | Fast-SCNN model definition |
| `loss/` | Training losses (e.g. Dice) |
| `utils/` | Dataset, transforms, training helpers |
| `paper/` | Paper figures (e.g. architecture diagram) |
| `train.py` | Training entry point |
| `test.py` | Example inference on images / video |

## Architecture

![network arch](paper/imgs/arch.png)

### Table 1 — Network structure

| Input | Block | t | c | n | s |
|:-:|---|---|---|---|---|
| 1024 × 2048 × 3 | Conv2D | - | 32 | 1 | 2 |
| 512 × 1024 × 32 | DSConv | - | 48 | 1 | 2 |
| 256 × 512 × 48 | DSConv | - | 64 | 1 | 2 |
| 128 × 256 × 64 | bottleneck | 6 | 64 | 3 | 2 |
| 64 × 128 × 64 | bottleneck | 6 | 96 | 3 | 2 |
| 32 × 64 × 96 | bottleneck | 6 | 128 | 3 | 1 |
| 32 × 64 × 128 | PPM | - | 128 | - | - |
| 32 × 64 × 128 | FFM | - | 128 | - | - |
| 128 × 256 × 128 | DSConv | - | 128 | 2 | 1 |
| 128 × 256 × 128 | Conv2D | - | nums of classes | 1 | 1 |

### Table 2 — Operator notation

| Input | Operator | Output |
|:-:|---|---|
| h × w × c | Conv2D 1/1, f | h × w × tc |
| h × w × tc | DWConv 3/s, f | h/s × w/s × tc |
| h/s × w/s × tc | Conv2D 1/1, − | h/s × w/s × c′ |

## Usage

### Environment

Create a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** `requirements.txt` pins older PyTorch/CUDA builds. Adjust `torch` / `torchvision` to match your CUDA or CPU setup if install fails.

### Dataset layout

Point `--data-root` at the folder that contains your images, labels, and list files. Each line in `train.txt` / `val.txt` is: **image path** and **label path**, both **relative to `data-root`** (space-separated).

Example:

```text
my_dataset/
  train.txt
  val.txt
  img/
    image1.jpg
    image2.jpg
  label/
    image1.png
    image2.png
```

`train.txt` / `val.txt` (paths relative to `my_dataset/`):

```text
img/image1.jpg label/image1.png
img/image2.jpg label/image2.png
```

### Training

Single GPU (default device is CUDA if available, else CPU):

```bash
python train.py --data-root /path/to/my_dataset --num-classes 21
```

Common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--data-root` | `voc2012` | Dataset root |
| `--train-list` | `<data-root>/train.txt` | Training list file |
| `--val-list` | `<data-root>/val.txt` | Validation list file |
| `--num-classes` | `21` | Class count (e.g. 21 for VOC) |
| `--epochs` | `2000` | Total epochs |
| `--batch-size` | `160` | Train batch size |
| `--base-lr` | `0.01` | Initial learning rate |
| `--input-h`, `--input-w` | `320`, `320` | Crop size |
| `--save-dir` | `save` | Checkpoint directory |
| `--save-freq` | `20` | Save every N epochs |
| `--log-dir` | `runs` | TensorBoard logs |
| `--resume` | — | Checkpoint path to continue training |

Resume from a checkpoint:

```bash
python train.py --data-root /path/to/my_dataset --resume save/train_100.pth
```

TensorBoard:

```bash
tensorboard --logdir runs
```

Multi-GPU with `DataParallel`:

```bash
python train.py --data-root /path/to/my_dataset --multigpu
```

Distributed training (typical launcher; adjust for your PyTorch version):

```bash
python -m torch.distributed.launch --nproc_per_node=N train.py --data-root /path/to/my_dataset --dist
```

### Pretrained weights (optional)

A VOC2012-oriented checkpoint trained at 540×540 is shared for faster initialization (Chinese README link; extract password from original mirror if needed):

[百度网盘](https://pan.baidu.com/s/17_pGbpkI4tx8eOMZFS73fA) · password: `v98k`

### Inference (`test.py`)

`test.py` loads weights from `WEIGHTS_PATH` (default `save/train_1999.pth`), uses CUDA, and expects class count / resolution consistent with the script (`MDL_CLS`, `seg.hw`). Before running:

1. Place a compatible checkpoint at `save/train_1999.pth` (or edit `WEIGHTS_PATH` in `test.py`).
2. For video, ensure `test.mp4` exists or change the path in `if __name__ == '__main__'`.
3. Create an output folder if required (e.g. `./result/` for `processVideo`).

Then run:

```bash
python test.py
```

Adapt `testImg('your_image.jpg')` vs `processVideo('your_video.mp4')` in the `__main__` block for image vs video.

## TODO

- [x] Training & validation
- [x] TensorBoard logging
- [x] Resume training
- [x] VOC2012 training script
- [x] Multi-GPU training

## Support

If you want to support this project:

<a href="https://www.buymeacoffee.com/winafoxq"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=winafoxq&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" /></a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zacario-li/Fast-SCNN_pytorch&type=Date)](https://star-history.com/#zacario-li/Fast-SCNN_pytorch&Date)
