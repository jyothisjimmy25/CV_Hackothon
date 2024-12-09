# Wildlife Monitoring System 🐾

## Project Overview
The **Wildlife Monitoring System** is an advanced tool designed to detect and classify animals in images using **YOLOv8**. This project is built to assist in monitoring wildlife, studying animal behavior, and aiding conservation efforts by leveraging computer vision and deep learning.

---

## Features
- **Animal Detection:** Detects multiple animals in a single image.
- **Bounding Box Visualization:** Draws bounding boxes around detected animals with labels and confidence scores.
- **Streamlit Integration:** User-friendly web interface for uploading and analyzing images.
- **Customizable Confidence Threshold:** Adjust the sensitivity of detections through a slider.
- **Dataset Support:** Integrated with Kaggle's Animal Image Dataset containing 90 different animals.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/jyothisjimmy25/CV_Hackothon.git
cd CV_Hackothon
exit
```
---
## Install Dependencies
Ensure Python 3.8 or higher is installed. Run:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Streamlit App
1. Navigate to the project directory.
2. Execute the Streamlit app:
   ```bash
   streamlit run stream.py
```
3. Open the app in your browser at `http://localhost:8501`.
```
---

### Running YOLOv8 Detection Notebook
1. Open the Jupyter notebook `cv.ipynb`.
2. Follow the steps in the notebook to perform object detection on images from the dataset.

---

## How It Works
1. **Image Input:** Upload an image or use a random sample from the dataset.
2. **YOLOv8 Detection:** The YOLOv8 model identifies animals, draws bounding boxes, and calculates confidence scores.
3. **Output Display:** The results include annotated images and detailed detection statistics.

---

## Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - YOLOv8
  - Streamlit
  - OpenCV
  - Pandas
  - Pillow
- **Frameworks:**
  - Streamlit for the web interface
  - Jupyter Notebook for detailed analysis
- **Dataset:** [Animal Image Dataset](https://www.kaggle.com/iamsouravbanerjee/animal-image-dataset-90-different-animals)

---

## Challenges and Future Scope

### Challenges
- 🐾 Managing model accuracy across diverse animal species.
- 🔄 Optimizing performance for large datasets.

### Future Scope
- 📹 Adding real-time video detection.
- 🗺️ Integrating GPS data for wildlife tracking.
- 📱 Building a mobile-friendly app for field researchers.

