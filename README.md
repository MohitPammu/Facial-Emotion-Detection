# Facial Emotion Detection

![Facial Emotion Detection](https://img.shields.io/badge/Project-Deep%20Learning-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.x-green)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-82%25-brightgreen)

## Project Overview

This project implements a deep learning solution for facial emotion detection, capable of classifying facial expressions into four emotional categories: happy, sad, neutral, and surprise. Leveraging Convolutional Neural Networks (CNNs), the system achieves 82% accuracy on unseen test data, making it suitable for real-world applications in human-computer interaction, mental health monitoring, and customer analytics.

## Repository Structure

```
Facial-Emotion-Detection/
├── Facial-Emotion-Detection.ipynb          # Complete analysis & implementation
├── Facial-Emotion-Detection.html           # HTML version of notebook
├── models/                                 # Trained models and architecture files
├── requirements.txt                        # Project dependencies
├── README.md                               # Project documentation
├── Facial Emotion Detection Report.pdf      # Detailed technical report
└── Facial_Emotion_Detection_Slide_Deck.pdf      # Project presentation
```

## Business Impact

The global Affective Computing market, valued at $62.53 billion in 2023, is projected to expand at a CAGR of 30.6%, reaching $388.28 billion by 2030. This growth is driven by the integration of emotion-aware technologies across sectors such as healthcare, automotive, education, and customer service.

**Key Business Applications:**
- **Human-Computer Interaction:** Enhanced user experience through emotion-aware interfaces
- **Mental Health Monitoring:** Real-time emotional state assessment for therapeutic applications
- **Customer Analytics:** Emotion-based feedback analysis for retail and service industries
- **Automotive Safety:** Driver emotion monitoring for safety systems
- **Educational Technology:** Student engagement and emotional state tracking

## Problem Statement

As artificial intelligence continues to evolve, enabling machines to interpret and respond to human emotions—known as **Affective Computing**—has become a pivotal frontier. Facial expression recognition is central to this endeavor, given that over 55% of human emotional communication is conveyed through facial cues.

This project aims to develop a robust deep learning model capable of performing multi-class emotion classification with high accuracy. The model can classify facial expressions into distinct emotion categories using grayscale images.

## Dataset

The dataset consists of grayscale facial images organized into four emotion categories:
- **Happy**: Images of people with happy facial expressions
- **Sad**: Images of people with sad or upset facial expressions
- **Surprise**: Images of people with shocked or surprised facial expressions
- **Neutral**: Images of people showing no prominent emotion

The dataset is divided into three folders:
- `train`: Used for model training
- `validation`: Used for model validation during training
- `test`: Used for final model evaluation

Initial class distribution showed an imbalance, with the 'surprise' class having fewer samples. This was addressed through data augmentation techniques.

## Methodology

### Data Preprocessing
- **Data Exploration**: Analysis of class distribution, pixel intensity distributions, and visual features
- **Data Augmentation**: Applied to balance underrepresented classes, particularly 'surprise'
- **Normalization**: Pixel values rescaled to [0,1] range

### Models Implemented and Tested

Several models were implemented and evaluated:

#### Custom CNN Models
1. **Model 1 (Grayscale & RGB)**:
   - Basic CNN with three convolutional blocks
   - ~605K parameters
   - Performance: 66% test accuracy

2. **Model 2 (Grayscale & RGB)**:
   - Deeper CNN with four convolutional blocks and batch normalization
   - ~390K parameters
   - Performance: 72% test accuracy (grayscale), 66% (RGB)

3. **Model 3 (Grayscale)** - **Best Performing Model**:
   - Complex CNN with three convolutional blocks, dual convolutions per block
   - Advanced regularization and batch normalization
   - ~1.5M parameters
   - Performance: 82% test accuracy

#### Transfer Learning Models
1. **VGG16**:
   - Pre-trained on ImageNet, fine-tuned for emotion detection
   - ~173K trainable parameters
   - Performance: 51% test accuracy

2. **ResNet101**:
   - Pre-trained on ImageNet with additional dense layers
   - ~2.1M trainable parameters
   - Performance: 25% test accuracy

3. **EfficientNetV2B2**:
   - Pre-trained on ImageNet with custom classification head
   - ~1.3M trainable parameters
   - Performance: 25% test accuracy

### Training Configuration
- **Optimizer**: Adam with learning rate of 0.001
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction
- **Data Augmentation**: Horizontal flipping, rotation, brightness adjustment, zoom

## Results

### Performance Comparison

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Status |
|-------|---------------|---------------------|---------------|---------|
| Model 1 (Grayscale) | 65% | 66% | 66% | Baseline |
| Model 1 (RGB) | 65% | 66% | 66% | Baseline |
| Model 2 (Grayscale) | 75% | 70% | 72% | Improved |
| Model 2 (RGB) | 67% | 66% | 66% | Standard |
| **Model 3 (Grayscale)** | **78%** | **76%** | **82%** | **🏆 Best Performance** |
| VGG16 | 52% | 54% | 51% | Transfer Learning |
| ResNet101 | 25% | 24% | 25% | Transfer Learning |
| EfficientNetV2B2 | 25% | 24% | 25% | Transfer Learning |

The best-performing model (Model 3) achieved **82% accuracy** on the test set, with a balanced performance across all emotion classes.

### Key Insights

- **Grayscale vs. RGB**: Models trained on grayscale images consistently outperformed RGB counterparts, as the dataset consists of grayscale images.
- **Transfer Learning Limitations**: Pre-trained models performed poorly due to the mismatch between their RGB-trained weights and the grayscale nature of our dataset.
- **Class Confusion**: 'Sad' and 'neutral' classes were the most frequently confused due to their similar visual features.
- **Data Augmentation Effect**: Augmentation significantly improved model performance on underrepresented classes.

## Model Architecture (Best Performing)

Model 3 architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 48, 48, 32)        320       
batch_normalization (BatchNo (None, 48, 48, 32)        128       
conv2d_1 (Conv2D)            (None, 48, 48, 32)        9248      
batch_normalization_1 (Batch (None, 48, 48, 32)        128       
max_pooling2d (MaxPooling2D) (None, 24, 24, 32)        0         
dropout (Dropout)            (None, 24, 24, 32)        0         
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
batch_normalization_2 (Batch (None, 24, 24, 64)        256       
conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928     
batch_normalization_3 (Batch (None, 24, 24, 64)        256       
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
dropout_1 (Dropout)          (None, 12, 12, 64)        0         
conv2d_4 (Conv2D)            (None, 12, 12, 128)       73856     
batch_normalization_4 (Batch (None, 12, 12, 128)       512       
conv2d_5 (Conv2D)            (None, 12, 12, 128)       147584    
batch_normalization_5 (Batch (None, 12, 12, 128)       512       
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 128)         0         
dropout_2 (Dropout)          (None, 6, 6, 128)         0         
flatten (Flatten)            (None, 4608)              0         
dense (Dense)                (None, 256)               1179904   
batch_normalization_6 (Batch (None, 256)               1024      
dropout_3 (Dropout)          (None, 256)               0         
dense_1 (Dense)              (None, 128)               32896     
batch_normalization_7 (Batch (None, 128)               512       
dropout_4 (Dropout)          (None, 128)               0         
dense_2 (Dense)              (None, 4)                 516       
=================================================================
Total params: 1,503,076
Trainable params: 1,501,412
Non-trainable params: 1,664
_________________________________________________________________
```

## Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.5+
See requirements.txt for complete dependency list
```

### Installation
```bash
# Clone the repository
git clone https://github.com/MohitPammu/Facial-Emotion-Detection.git
cd Facial-Emotion-Detection

# Install required packages
pip install -r requirements.txt

# Open the comprehensive analysis
jupyter notebook Facial-Emotion-Detection.ipynb
```

### Quick Start
```python
# Load the trained model for inference
from tensorflow.keras.models import load_model
model = load_model('models/model_3.keras')

# Predict emotions on new images
predictions = model.predict(new_images)
```

## Production Deployment Considerations

- **Edge Computing:** Optimized architecture suitable for real-time applications
- **API Integration:** Model ready for REST API deployment
- **Privacy Compliance:** Designed for local processing to maintain data privacy
- **Scalability:** Efficient inference for high-throughput scenarios

## Challenges and Limitations

- **Class Imbalance**: Despite augmentation, class distribution remains a challenge.
- **Emotion Ambiguity**: Confusion between visually similar emotions (e.g., sad vs. neutral).
- **Static Images**: The model analyzes static images, limiting its ability to capture temporal aspects of emotions.
- **Dataset Size**: Limited dataset size may affect model generalization to diverse populations.

## Recommendations for Implementation

- Deploy Model 3 (grayscale CNN) for production use due to its superior performance.
- Implement continuous model monitoring and retraining with new data.
- Consider integrating temporal analysis (video sequences) for improved accuracy.
- Expand dataset diversity to enhance model generalization across demographics.
- Implement ethical safeguards and compliance with privacy regulations.

## Future Work

- Implement multi-modal emotion recognition by incorporating voice and text analysis.
- Explore more complex architectures such as attention-based networks.
- Extend the emotion classes to include more nuanced categories.
- Develop real-time deployment solutions for edge devices.
- Conduct user acceptance testing in real-world scenarios.

## Project Context

This project was developed as part of the MIT Professional Education Applied Data Science certification program, demonstrating practical application of deep learning techniques for computer vision and affective computing applications.

## Portfolio

Explore my complete data science portfolio at [mohitpammu.github.io/Projects](https://mohitpammu.github.io/Projects/)

## Connect

- **LinkedIn:** [mohitpammu](https://linkedin.com/in/mohitpammu)
- **Email:** mopammu@gmail.com

## Acknowledgments

- MIT Professional Education Applied Data Science Program
- The creators of the facial emotion datasets
- Contributors to TensorFlow and Keras frameworks

---

*Developed by Mohit Pammu, MBA | Experienced professional entering data science with production-ready skills*
