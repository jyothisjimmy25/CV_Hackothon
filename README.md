Wildlife Monitoring System 🐾
Project Overview
The Wildlife Monitoring System is an advanced tool designed to detect and classify animals in images using YOLOv8. This project is built to assist in monitoring wildlife, studying animal behavior, and aiding conservation efforts by leveraging computer vision and deep learning.

Features
Animal Detection: Detects multiple animals in a single image.
Bounding Box Visualization: Draws bounding boxes around detected animals with labels and confidence scores.
Streamlit Integration: User-friendly web interface for uploading and analyzing images.
Customizable Confidence Threshold: Adjust the sensitivity of detections through a slider.
Dataset Support: Integrated with Kaggle's Animal Image Dataset containing 90 different animals.
Installation
Clone the Repository

bash
Copy code
git clone <repository-link>
cd wildlife-monitoring-system
Install Dependencies Ensure Python 3.8 or higher is installed. Run:

bash
Copy code
pip install -r requirements.txt
Download YOLOv8 Model

Download the YOLOv8 model from the Ultralytics YOLO Repository.
Place the model (yolov8n.pt) in the root directory.
Install Streamlit

bash
Copy code
pip install streamlit
Usage
Running the Streamlit App
Navigate to the project directory.
Execute the Streamlit app:
bash
Copy code
streamlit run stream.py
Open the app in your browser at http://localhost:8501.
Running YOLOv8 Detection Notebook
Open the Jupyter notebook cv.ipynb.
Follow the steps in the notebook to perform object detection on images from the dataset.
How It Works
Image Input: Upload an image or use a random sample from the dataset.
YOLOv8 Detection: The YOLOv8 model identifies animals, draws bounding boxes, and calculates confidence scores.
Output Display: The results include annotated images and detailed detection statistics.
Tech Stack
Python Libraries:
YOLOv8
Streamlit
OpenCV
Pandas
Frameworks: Streamlit for the web interface, Jupyter for notebook analysis.
Dataset: Animal Image Dataset.
Screenshots
Streamlit Web App


Detection Results


Challenges and Future Scope
Challenges
Managing model accuracy across diverse animal species.
Optimizing performance for large datasets.
Future Scope
Adding real-time video detection.
Integrating GPS data for wildlife tracking.
Building a mobile-friendly app for field researchers.
