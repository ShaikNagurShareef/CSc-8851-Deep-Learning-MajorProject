# CSc-8851-Deep-Learning-MajorProject
CSc 8851: Deep Learning, Fall 2024, Final Project Code

# VL-VAE MedICaT :: Vision-Language Variational Autoencoder based Medical Image Captioning Transformer

## Overview
This repository implements a Vision-Language Variational Autoencoder (VL-VAE) tailored for medical image captioning. The model integrates multi-modal data (images and text) using advanced neural network techniques, providing a framework for generating meaningful captions for medical images while maintaining latent feature alignment.

## Features
- **Multi-modal learning:** Joint processing of visual (images) and textual (keywords, captions) data.
- **Vision-Language VAE:** A hybrid approach combining image and text encoders with a shared latent space.
- **Custom loss functions:** Includes VAE loss, orthogonality constraints, and shared information alignment loss.
- **Attention mechanisms:** Utilizes global context attention and attention gate mechanisms.
- **Pre-trained backbones:** Supports efficient feature extraction with models like EfficientNet.
- **Comprehensive training pipeline:** Includes data preprocessing, augmentation, and evaluation.

## Requirements
The script relies on the following key libraries:
- Python
- PyTorch
- Torchvision
- Timm
- Scikit-learn
- Matplotlib
- Pandas
- PIL
- Rouge
- NLTK
- BertScore
- Thop

## Modules

### 1. **Configuration (`Config` Class)**
Handles configuration settings for the entire pipeline, including dataset paths, model hyperparameters, and training parameters. Key parameters include:
- `device`: Hardware to use (CPU/GPU).
- `image_dir`: Directory containing image data.
- `latent_dim`: Dimension of the shared latent space.
- `epochs`, `batch_size`, `lr`: Training hyperparameters.

### 2. **Data Processing**
- **Data Preparation:** Loads JSON files with image paths, keywords, and descriptions into Pandas DataFrames.
- **Image Transformations:** Applies resizing, normalization, and other augmentations.
- **Tokenizer:** Processes textual data into tokenized formats and builds a vocabulary.

### 3. **Custom Dataset (`DeepEyeNetDataset` Class)**
Defines a dataset class for handling medical images and corresponding text descriptions. Features include:
- Image loading and transformation.
- Tokenization and padding of text.

### 4. **Model Architecture**
#### a. **Image Encoder**
Uses EfficientNet for feature extraction. Includes attention blocks (Global Context Attention, Attention Gate) for enhanced feature learning.

#### b. **Text Encoder**
Implements a transformer-based encoder with sinusoidal positional encodings.

#### c. **Vision-Language VAE**
- Encodes images and text into a shared latent space.
- Includes separate and joint latent representations for images and text.
- Dynamically builds decoders for reconstruction.

#### d. **Fusion Mechanisms**
- **TransFusion Encoder:** Combines features from image and text modalities.
- **Latent TransFusion Encoder:** Aligns latent features from different modalities.

#### e. **Transformer Decoder**
Generates captions from fused features using a multi-layer transformer with attention.

### 5. **Loss Functions**
- **VL-VAE Loss:** Includes reconstruction and KL divergence losses.
- **Orthogonality Constraint Loss:** Enforces orthogonality between modality-specific features.
- **Shared Information Alignment Loss:** Aligns latent embeddings across modalities.

### 6. **Training & Evaluation**
- Implements a complete pipeline with data loaders, optimizer setup, and checkpointing.
- Tracks training and validation losses.
- Evaluation includes BLEU, ROUGE, and BERTScore metrics for caption quality.

### 7. **Utilities**
- **FLOP Computation:** Calculates floating-point operations per second (FLOPs) for model efficiency.
- **Caption Generation:** Provides a function to generate captions for given images and keywords.
- **Visualization:** Plots training curves and displays sample generated captions.

## Usage

### Training
1. Place your dataset in the specified directories.
2. Configure paths and hyperparameters in the `Config` class.
3. Run the script to start training.

```bash
python VL-VAE_MedICaT.py
```

### Evaluation
To evaluate the model on the validation set:
```bash
python VL-VAE_MedICaT.py --evaluate
```

### Outputs
- **Model Checkpoints:** Saved in the specified path in `Config`.
- **Generated Captions:** Stored in `results.csv`.
- **Training Curves:** Saved as `acc_loss_plot.png`.

## Metrics
- **BLEU:** Measures n-gram overlaps.
- **ROUGE:** Evaluates recall-oriented overlap.
- **BERTScore:** Assesses semantic similarity.

## Future Work
- Extend support for additional pre-trained models.
- Experiment with other loss functions and attention mechanisms.
- Add support for other datasets and multi-lingual captions.

## Authors
- **Nagur Shareef Shaik**
- **Teja Krishna Cherukuri**

  Note: Equal contribution by both.

## Acknowledgments
Developed by leveraging state-of-the-art techniques in AI and medical image analysis. Special thanks to the open-source community for pre-trained models and evaluation tools.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.


