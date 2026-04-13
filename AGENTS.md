## Cursor Cloud specific instructions

### Project overview
Fast-SCNN is a PyTorch implementation of the Fast Semantic Segmentation Network. See `readme.md` for architecture details.

### Key entry points
- `train.py` — trains on a dataset (defaults to VOC2012, 21 classes, 320x320 input)
- `test.py` — runs inference on video/images (requires trained weights at `save/train_1999.pth`)
- `models/fastscnn.py` — model definition; runnable standalone as an FPS benchmark

### Dependencies
The `requirements.txt` pins very old versions (2020-era) that are incompatible with Python 3.12. Install modern versions instead:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python-headless tensorboardX tqdm
```
- `apex` in `requirements.txt` is **not imported** anywhere and can be safely skipped.
- Use `opencv-python-headless` (not `opencv-python`) in headless/CI environments.

### Known compatibility issues
- `utils/transform.py` uses `collections.Iterable`, which was removed in Python 3.10+. The fix is `collections.abc.Iterable`. This affects data augmentation transforms used during training.
- `utils/common.py` `intersectionAndUnionGPU()` calls `.cuda()` unconditionally. On CPU-only environments, training/validation will fail at that call.
- `test.py` calls `.cuda()` directly and requires a GPU with trained weights to run.

### Running without a GPU
The model itself (`FastSCNN`) works on CPU. You can instantiate and do forward passes:
```python
from models.fastscnn import FastSCNN
model = FastSCNN(21, aux=True)
model.eval()
out = model(torch.randn(1, 3, 320, 320))
```
Full training (`train.py`) and inference (`test.py`) scripts assume CUDA and a dataset/weights on disk.

### No linter or test suite
This repository has no configured linter, formatter, or automated test suite.
