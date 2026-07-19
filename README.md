# FaceMatcher - Person Re-Identification Desktop Application

FaceMatcher is a desktop application that finds visually similar people across images, videos and webcams using **Person Re-Identification (ReID)** technology.

Unlike traditional facial recognition systems, FaceMatcher matches people based on their overall appearance, including clothing, body shape, colors, and visual features. This makes it useful in situations where faces are partially visible, turned away, blurred, or too small for reliable facial recognition.

| Reference Person                             | Selected Source                           | Matching Result                           |
| :------------------------------------------: | :---------------------------------------: | :---------------------------------------: |
| <img height="300" alt="Referenc Image" src="https://github.com/John-Da/FaceMatcher/blob/main/reference.png"> | <img width="500" alt="crossRoad" src="https://github.com/John-Da/FaceMatcher/blob/main/crossRoad.jpg"> | <img width="500" alt="result" src="https://github.com/John-Da/FaceMatcher/blob/main/result.png"> |

## Features

* 🔍 Search for a person using a reference image
* 🎥 Match people across videos
* 📷 Real-time webcam matching
* 📁 Search through image folders
* 📊 Similarity score visualization
* 📝 Match logging and result tracking
* 🖥️ User-friendly desktop interface built with PySide6
* ⚡ Fast inference using modern Person ReID models

## How It Works

1. Select a reference image containing the target person.
2. Choose an input source:

   * Image
   * Video
   * Webcam
   * Folder
3. FaceMatcher extracts appearance features from the reference person.
4. The system compares these features against detected people in the selected source.
5. Matching results are displayed with similarity scores and bounding boxes.

## Use Cases

* Security and surveillance research
* Person retrieval in large image collections
* Video investigation workflows
* AI and computer vision demonstrations
* ReID model evaluation and testing

## Technology Stack

* Python
* PySide6
* OpenCV
* PyTorch
* YOLO26s (Person Detection)
* OSNet-x1 (Person Re-Identification)
* BoxMOT Occluboost (Tracking + Cosine Similarity Filtering)


## 📺 APP Demo

https://github.com/user-attachments/assets/48f9dd4a-5ba7-4b2d-8e85-2f458c582236

<br>

## ⚠️ Known Limitations

> FaceMatcher relies on appearance-based Person Re-Identification rather than facial recognition. Matching performance is heavily dependent on the capabilities of the underlying ReID model.

Current limitations include:

* Similar clothing between different individuals may lead to false matches.
* Significant changes in clothing, accessories, or appearance can reduce matching accuracy.
* Extreme occlusion, poor lighting, low resolution, or unusual camera angles may affect results.
* Re-identification performance is limited by the capabilities of the OSNet-x1 model.
* Similarity scores should be treated as confidence indicators rather than definitive identity verification.

### Development Status

The following features are currently under active development and may be unstable or incomplete:

* Video processing workflow
* Real-time webcam matching
* Advanced tracking and timeline visualization
* Match logging and analytics improvements


## Installation

```bash
git clone https://github.com/yourusername/FaceMatcher.git

cd FaceMatcher

python -m venv venv

source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Project Status

FaceMatcher is currently under active development: new features like video and webcame and performance improvements.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/John-Da/FaceMatcher/blob/main/LICENSE) file for details.

