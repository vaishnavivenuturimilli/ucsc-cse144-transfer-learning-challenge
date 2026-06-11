# ucsc-cse144-transfer-learning-challenge

**Author:** Vaishnavi Venuturimilli  
**Kaggle Public Leaderboard Score:** 65.45%  

## Project Overview
The goal of this project was to build an image classifier capable of categorizing images into 100 different classes under a strict data constraint of only 10 training images per class. 
The final architecture utilizes a ResNet-50 backbone with a custom classifier head, heavily regularized using a 0.5 Dropout rate, an AdamW optimizer with a 0.1 weight decay, and a StepLR scheduler.

## Trained Model Weights
Due to GitHub's file size limits, the trained PyTorch weights for the final model (`vaishnavi_cse144_finalmodel.pth`) are hosted externally. 
* **Download the weights here:** 

## Environment Setup
Training and inference were conducted locally utilizing PyTorch's `mps`  backend for Apple Silicon hardware acceleration. The code is hardcoded to `device = torch.device("mps")`, but can be easily modified to `"cuda"` or `"cpu"` depending on your hardware.

**Dependencies:**
* Python 3.11.8
* `torch` (Tested on v2.12.0)
* `torchvision`
* `pandas`
* `numpy`
* `matplotlib`
* `tqdm`
* `Pillow` (PIL)

## How to Reproduce Results
1. Clone this repository to your local machine.
2. Download the Kaggle dataset and unzip it into the root directory. Ensure the images are inside relative paths named `train/` and `test/`.
3. Download the trained model weights from the Google Drive link above and place `vaishnavi_cse144_finalmodel.pth` in the root directory.
4. Open the provided Jupyter Notebook (`.ipynb`).
5. **To Train:** Run the training cell.
6. **To Evaluate:** Run the inference cell. It will automatically load the saved `.pth` weights, evaluate the `test/` directory, and output the properly formatted `submission.csv`.
