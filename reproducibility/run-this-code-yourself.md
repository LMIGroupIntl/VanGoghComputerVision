# Run this code yourself

## Reproducing the Training

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/VanGoghComputerVision.git
cd VanGoghComputerVision
```

2. Prepare your dataset

* Start with the 'images.csv' file from the [`data/`](https://github.com/lmigroupintl/VanGoghComputerVision/tree/main/data) folder
* Organize the images into the following directory structure:

```
dataset/
├── authentic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── imitation/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Training Models

4. Run unoptimized training (baseline)

```bash
python scripts/train_unoptimized.py --data_dir /path/to/the/dataset
```

5. Run optimized training (regularized Swin + stable EfficientNet

```bash
python scripts/train_optimized.py --data_dir /path/to/the/dataset
```

## Running Inference

6. **Inference with all models in a directory:**

```bash
python scripts/inference.py --models /path/to/models/directory --images /path/to/test/images
```

7. Inference with a specific model:

```bash
python scripts/inference.py --models /path/to/specific/model --images /path/to/test/images
```

8. Single image inference:

```bash
python scripts/inference.py --models /path/to/models --images single_image.jpg
```

9. Inference with majority/mean aggregation (default is majority voting):

```bash
python scripts/inference.py --models /path/to/models --images /path/to/test/images --aggregation mean
```

## Jupyter Notebooks (Alternative)

10. Launch Jupyter notebooks for interactive exploration:

```bash
jupyter notebook notebooks/
```

The training scripts will automatically:

* Extract patches using adaptive grid sizes (1x1, 2x2, or 4x4 based on image dimensions)
* Apply appropriate regularization techniques
* Save trained models and metrics to the output directory
* Generate training progress plots
