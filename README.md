# PyTorch SRGAN for SISR task

This project implements a **SRGAN** (Super-Resolution Generative Adversarial Network) model for **SISR** (Single Image Super-Resolution) task. The primary goal is to upscale low-resolution (LR) images by a given factor (2x, 4x, 8x) to produce super-resolution (SR) images with high fidelity and perceptual quality.

This implementation is based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) and inspired by the [sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution).

## Demonstration

The following images compare the standard bicubic interpolation with the output of the SRGAN model.

![Baboon comparison image](images/comparison_baboon.png)
![Butterfly comparison image](images/comparison_butterfly.png)
![Bird comparison image](images/comparison_bird.png)
![Man comparison image](images/comparison_man.png)
![PPT3 comparison image](images/comparison_ppt3.png)

## Key Features

- Uses **PixelShuffle** instead of **TransposedConvolution** for cleaner upscaling and fewer artifacts.
- Leverages **autocast** and **GradScaler** for significant training speedup on compatible GPUs.
- Applies random rotations (90/180/270 degrees) and flips (horizontal/vertical) during training to improve model generalization.
- The `SRDataset` class can load images directly from a directory path or from a `.txt` file listing image paths.
- Calculates PSNR and SSIM on the **Y-channel** (luminance) after RGB conversion, adhering to the standard SR evaluation practice.
- Implements pixel shaving (cropping by scaling factor) on boundaries before metric calculation to avoid border effects.
- Automatically tracks the best content loss (VGG loss) and saves this model state separately to `checkpoints/srgan_best`.
- Uses the `.safetensors` format instead of pickle (`.pth`) for saving weights, which is safer and faster.
- All hyperparameters, paths, and settings are managed in a single `config.py` file for easy experiment management.
- Saves the latest checkpoint after every epoch and saves the current state on **KeyboardInterrupt** (Ctrl+C).
- All training progress is logged to both the console and a timestamped log file in the `logs/` directory.
- Automatically generates and saves training plots (Loss, PSNR, SSIM) upon completion.

## Model Architectures

### Generator

As a Generator, this project uses pretrained **SRResNet** model (500 epochs) from my [previous project](https://github.com/ash1ra/PyTorch-SRResNet-SISR-COCO2017).

```ascii
                                     Input (LR Image)
                                            |
                                            v
                        +-Input-Conv-Block-----------------------+
                        | Conv2D (9x9 kernel) (3 -> 64 channels) |
                        | PReLU                                  |
                        +----------------------------------------+
                                            |
                                            +---------------------------+
                                            |                           |
                                            v                           |
                  +-----+-16x-Residual-Blocks---------------------+     |
                  |     | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                  |     | Batch Normalization                     |     |
(Skip connection) |     | PReLU                                   |     | (Skip connection)
                  |     | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                  |     | Batch Normalization                     |     |
                  +-----+-----------------------------------------+     |
                                            |                           |
                                            v                           |
                        +-Middle-Conv-Block-----------------------+     |
                        | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                        | Batch Normalization                     |     |
                        +-----------------------------------------+     |    
                                            |                           |
                                            +---------------------------+
                                            |
                                            v
                        +-2x-Sub-pixel-Conv-Blocks-----------------+
                        | Conv2D (3x3 kernel) (64 -> 256 channels) |
                        | PixelShuffle (h, w, 256 -> 2h, 2w, 64)   |
                        | PReLU                                    |
                        +------------------------------------------+
                                            |
                                            v
                        +-Final-Conv-Block-----------------------+
                        | Conv2D (9x9 kernel) (64 -> 3 channels) |
                        | Tanh                                   |
                        +----------------------------------------+
                                            |
                                            v
                                     Output (SR Image)
```

### Discriminator

As a Discriminator, this project uses a convolutional neural network that functions as a binary image classifier.

***Note:*** *The result of the model is logit, which is then passed to `BCEWithLogitsLoss` (with built-in `Sigmoid`) loss function, and therefore does not need a separate `Sigmoid` layer.*

```ascii
                                  Input (SR or HR Image)
                                            |
                                            v
                        +-Input-Conv-Block-----------------------+
                        | Conv2D (3x3 kernel) (3 -> 64 channels) |
                        | LeakyReLU                              |
                        +----------------------------------------+
                                            |
                                            v
                      +-7x-Conv-Blocks-(i=block-number)-------------+
                      | if i is odd: stride=2  | channels: C -> C   |
                      | if i is even: stride=1 | channels: C -> 2*C |
                      +---------------------------------------------+
                      | Conv2D (3x3 kernel)                         |
                      | Batch Normalization                         | 
                      | LeakyReLU                                   |
                      +---------------------------------------------+
                                            |                           
                                            v
                        +-Final-Block----------------------------+
                        | AdaptiveAvgPool2D (6x6)                |
                        | Flatten                                |
                        | Linear (512 * 6 * 6 -> 1024 channels)  |
                        | LeakyReLU                              |
                        | Linear (1024 -> 1 channel)             |
                        +----------------------------------------+
                                            |
                                            v
         Output (logit of probability of the original input being a natural image)
```

## Datasets

### Training

The model is trained on the **DF2K_OST** (DIV2K + Flickr2K + OST) dataset. The `data_processing.py` script dynamically creates LR images from HR images using bicubic downsampling and applies random crops and augmentations (flips, rotations).

### Validation

The **DIV2K_valid** dataset is used for validation.

### Testing

The `test.py` script is configured to evaluate the trained model on standard benchmark datasets: **Set5**, **Set14**, **BSDS100**, and **Urban100**.

## Project Structure

```
.
├── checkpoints/             # Stores model weights (.safetensors) and training states
├── images/                  # Directory for inference inputs, outputs, and training plots
├── config.py                # Configures the application logger, hyperparameters and file paths
├── data_processing.py       # Defines the SRDataset class and image transformations
├── inference.py             # Script to run the model on a single image
├── models.py                # Generator, Discriminator and TruncatedVGG19 model architectures definition
├── test.py                  # Script for evaluating the model on benchmark datasets
├── train.py                 # Script for training the model
└── utils.py                 # Utility functions (metrics, checkpoints, plotting)
```

## Configuration

All hyperparameters, paths, and training settings can be configured in the `config.py` file.

Explanation of some settings:
- `INITIALIZE_WITH_SRRESNET_CHECKPOINT`: Set to `True` to use pre-trained SRResNet weights.
- `LOAD_CHECKPOINT`: Set to `True` to resume training from the last SRGAN checkpoint.
- `LOAD_BEST_CHECKPOINT`: Set to `True` to resume training from the best SRGAN checkpoint.
- `TRAIN_DATASET_PATH`: Path to the train data. Can be a directory of images or a `.txt` file listing image paths.
- `VAL_DATASET_PATH`: Path to the validation data. Can be a directory of images or a `.txt` file listing image paths.
- `TEST_DATASETS_PATHS`: List of paths to the test data. Each path can be a directory of images or a `.txt` file listing image paths.
- `DEV_MOVE`: Set to `True` to use a 10% subset of the train data for quick testing.

***Note:*** *`INITIALIZE_WITH_SRRESNET_CHECKPOINT` and `LOAD_CHECKPOINT` or `LOAD_BEST_CHECKPOINT` are mutually exclusive. If the first one is `True`, then the other two must be `False` and vice versa.*

## Setting Up and Running the Project

### 1. Installation

1. Clone the repository:
```bash
git clone https://github.com/ash1ra/PyTorch-SRGAN-SISR-DF2K_OST.git
cd PyTorch-SRGAN-SISR-DF2K_OST
```

2. Create `.venv` and install dependencies:
```bash
uv sync
```

3. Activate a virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

### 2. Data Preparation

1.  [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) the **DIV2K** datasets (`Train Data (HR images)` and `Validation Data (HR images)`).
2.  [Download](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) the **Flickr2K** dataset.
3.  [Download](https://drive.google.com/drive/folders/1LIb631GU3bOyQVTeuALesD8_eoApNniB) the **OST** datasets (`OutdoorSceneTest300/OST300_img.zip` and `OutdoorSceneTrain_v2`).
4.  [Download](https://figshare.com/articles/dataset/BSD100_Set5_Set14_Urban100/21586188) the standard benchmark datasets (**Set5**, **Set14**, **BSDS100**, **Urban100**).
5.  Create training dataset from **DIV2K**, **Flickr2K** and **OST** (both, test and train).
6.  Organize your data directory as expected by the scripts:
    ```
    data/
    ├── DF2K_OST/
    │   ├── 1.jpg
    │   └── ...
    ├── DIV2K_valid/
    │   ├── 1.jpg
    │   └── ...
    ├── Set5/
    │   ├── baboon.png
    │   └── ...
    ├── Set14/
    │   └── ...
    ...
    ```

    or
    
    ```
    data/
    ├── DF2K_OST.txt
    ├── DIV2K_valid.txt
    ├── Set5.txt
    ├── Set14.txt
    ...
    ```
    
8.  Update the paths (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `TEST_DATASETS_PATHS`) in `config.py` to match your data structure.

### 3. Training

1.  Adjust parameters in `config.py` as needed.
2.  Run the training script:
    ```bash
    python train.py
    ```
3.  Training progress will be logged to the console and to a file in the `logs/` directory.
4.  Checkpoints will be saved in `checkpoints/`. A plot of the training metrics will be saved in `images/` upon completion.

### 4. Testing

To evaluate the model's performance on the test datasets:

1.  Ensure the `BEST_CHECKPOINT_DIR_PATH` in `config.py` points to your trained model (e.g., `checkpoints/srgan_best`).
2.  Run the test script:
    ```bash
    python test.py
    ```
3.  The script will print the average PSNR and SSIM for each dataset.

### 5. Inference

To upscale a single image:

1.  Place your image in the `images/` folder (or update the path).
2.  In `config.py`, set `INFERENCE_INPUT_PATH` to your image, `INFERENCE_OUTPUT_PATH` to desired location of output image, `INFERENCE_COMPARISON_IMAGE_PATH` to deisred location of comparison image (optional) and `BEST_CHECKPOINT_DIR_PATH` to your trained model.
3.  Run the script:
    ```bash
    python inference.py
    ```
4.  The upscaled image (`sr_img_*.png`) and a comparison image (`comparison_img_*.png`) will be saved in the `images/` directory.

## Training Results

![The following chart shows the progression of loss, learning rate, PSNR, and SSIM during training.](images/training_metrics_2025-11-04-01-41-51.png)

The model was trained for 500 epochs with a batch size of 32 on an NVIDIA RTX 4060 Ti (8 GB) and took nearly 15 hours. The rest of the hyperparameters are specified on the chart. The final model is the one with the lowest validation loss (content loss / VGG loss) value.

## Benchmark Evaluation (4x Upscaling)

The final model (`srgan_best`) was evaluated on standard benchmark datasets. Metrics are calculated on the Y-channel after shaving 4px (the scaling factor) from the border.

The results are compared against the original paper's SRGAN and the sgrvinod tutorial implementation.

**PSNR (dB) Comparison**
| Dataset / Implementation | SRGAN (this project) | SRGAN (sgrvinod) | SRGAN (paper)
| :--- | :---: | :---: | :---: |
| **Set5** | 28.7163 | 29.719 | 29.40 |
| **Set14** | 24.1836 | 26.509 | 26.02 |
| **BSDS100** | 23.2771 | 25.531 | 25.16 |
| **Urban100**| 21.9942 | — | — |

**SSIM Comparison**
| Dataset / Implementation | SRGAN (this project) | SRGAN (sgrvinod) | SRGAN (paper)
| :--- | :---: | :---: | :---: |
| **Set5** | 0.8339 | 0.859  | 0.8472 |
| **Set14** | 0.6919 | 0.729 | 0.7397 |
| **BSDS100** | 0.6426 | 0.678 | 0.6688 |
| **Urban100** | 0.7163 | — | — |

***Note:*** *My results might be slightly different from the paper's, which is expected. The paper's authors may have used different training datasets, different training durations, or minor variations in implementation.*

***Note 2:*** *It's important to remember that in Super-Resolution, traditional metrics like PSNR and SSIM are not the only measure of success. As highlighted in the tutorial and the original paper, a model (like SRResNet) trained to minimize MSE will maximize PSNR, but this often results in overly smooth images that lack fine, realistic textures. Perceptually-driven models (like SRGAN) often score lower on PSNR/SSIM but produce results that look much more convincing to the human eye.*

## Visual Comparisons

The following images compare the standard bicubic interpolation with the output of the SRGAN model. I tried to use different images that would be visible difference in results with anime images, photos etc.

![Comparisson image 1](images/comparison_img_1.png)
![Comparisson image 2](images/comparison_img_2.png)
![Comparisson image 3](images/comparison_img_3.png)
![Comparisson image 4](images/comparison_img_4.png)
![Comparisson image 5](images/comparison_img_5.png)

## Acknowledgements

This project is heavily inspired by the excellent [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution) by [sgrvinod](https://github.com/sgrvinod), which is based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

```bibtex
@misc{ledig2017photorealisticsingleimagesuperresolution,
      title={Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network}, 
      author={Christian Ledig and Lucas Theis and Ferenc Huszar and Jose Caballero and Andrew Cunningham and Alejandro Acosta and Andrew Aitken and Alykhan Tejani and Johannes Totz and Zehan Wang and Wenzhe Shi},
      year={2017},
      eprint={1609.04802},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1609.04802}, 
}
```

DIV2K dataset citation:

```bibtex
@InProceedings{Timofte_2018_CVPR_Workshops,
  author = {Timofte, Radu and Gu, Shuhang and Wu, Jiqing and Van Gool, Luc and Zhang, Lei and Yang, Ming-Hsuan and Haris, Muhammad and others},
  title = {NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


