## Automated Detection of Myocardial Infarction in Low-Quality Echocardiography

## Overview
This project automates the detection of myocardial infarction (MI) using deep learning on low-quality echocardiogram images. It includes segmentation and classification models to identify infarcted regions and classify echocardiograms.

## Key Features
- **Segmentation**: Attention U-Net for segmenting infarcted regions.
- **Classification**: Multiple classifiers for healthy vs MI cases.
- **Custom Loss**: Focal Tversky Loss for improved segmentation.
- **Metrics**: Dice Coefficient, IoU, Precision, Recall, Pixel-wise Accuracy.

## Project Structure
- `seghelper.ipynb`, `segtask.ipynb`: Segmentation pipeline and model training.
- `classifier-mi*.ipynb`: Classifier experiments and results.
- `featureextr.ipynb`: Feature extraction.
- `convlstm.ipynb`: ConvLSTM experiments.
- `Final_Presentation_MI.pdf`: Project summary and results.

## Data
- **Input**: Low-quality echocardiogram frames and masks.
- **Directories**: Update paths in notebooks to your local dataset.

## Usage
1. **Setup**: Install required Python packages (see notebook imports).
2. **Segmentation**: Run `segtask.ipynb` to train and evaluate segmentation.
3. **Classification**: Use `classifier-mi*.ipynb` for classification.
4. **Metrics**: Evaluate performance using provided metric functions.

## Results
- Robust segmentation and classification on low-quality data.
- See `Final_Presentation_MI.pdf` for results and analysis.

