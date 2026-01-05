# Speech Emotion Recognition

A machine learning and deep learning project for recognizing human emotions from speech audio files. This system analyzes acoustic features of speech signals to classify emotions such as happy, sad, angry, fearful, neutral, and more.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Speech Emotion Recognition (SER) is the task of identifying human emotions from speech. This project implements various machine learning and deep learning models to classify emotions from audio recordings. The system extracts acoustic features from speech signals and uses them to train classification models.

**Key Objectives:**
- Accurate classification of emotions from speech audio
- Extract meaningful audio features using signal processing methods such as MFCCs, chroma features, spectral contrast.
- Implement and compare Support Vector Machine (SVM) and Logistic Regression models to analyze classification performance.

## Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset:

- **Total Files**: 1,440 audio files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Statements**: 2 lexically-matched statements
- **Intensities**: Normal and strong (except neutral)

### Filename Convention

RAVDESS files follow this naming convention:
```
Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
```

**Example**: `03-01-05-02-01-01-12.wav`
- Modality: 03 (audio-only)
- Vocal Channel: 01 (speech)
- Emotion: 05 (angry)
- Intensity: 02 (strong)
- Statement: 01 ("Kids are talking by the door")
- Repetition: 01 (1st repetition)
- Actor: 12 (Actor #12)

### Emotion Labels

| Code | Emotion |
|------|---------|
| 01   | Neutral |
| 02   | Calm    |
| 03   | Happy   |
| 04   | Sad     |
| 05   | Angry   |
| 06   | Fearful |
| 07   | Disgust |
| 08   | Surprised |

**Dataset Link**: [RAVDESS on Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tejasviravikumar/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

```
numpy
pandas
librosa
scikit-learn
matplotlib
seaborn
glob
Ipython
```
### Feature Extraction

The system extracts the following acoustic features:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - Represents the short-term power spectrum of sound
   - Default: 40 coefficients

2. **Chroma Features**
   - Represents pitch class information
   - 12-dimensional feature vector

3. **Mel-Spectrogram**
   - Frequency representation on mel scale
   - Better aligned with human auditory perception

4. **Spectral Contrast**
   - Zero Crossing Rate
   - Spectral Centroid
   - Spectral Rolloff

### Model Options

The project implements multiple models:

1. **Traditional ML Classifiers**
   - Logistic Regression
   - SVM (Support Vector Machine)

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Machine learning algorithms and metrics
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## Project Structure

```
Speech-Emotion-Recognition/
├── Model/                      
├── notebooks/                 
├── src/                        
│   ├── data_preprocessing.py   
│   ├── feature_extraction.py  
│   ├── model.py               
│   └── utils.py                
├── data/                       
├── results/                    
├── requirements.txt            
└── README.md                   
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Implementing new model architectures
- Adding support for additional datasets
- Improving feature extraction techniques
- Enhancing real-time prediction capabilities
- Adding data augmentation methods
- Improving documentation

## 📧 Contact

**Tejasvi Ravikumar** - [@tejasviravikumar](https://github.com/tejasviravikumar)

Project Link: [https://github.com/tejasviravikumar/Speech-Emotion-Recognition](https://github.com/tejasviravikumar/Speech-Emotion-Recognition)

---

⭐ If you find this project helpful, please consider giving it a star!
