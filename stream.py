# import kagglehub
# import streamlit as st
# import glob
# import random
# import os
# import cv2
# from ultralytics import YOLO
# from PIL import Image
# import tempfile

# # Debug: Ensure Streamlit is loading properly
# st.write("Streamlit app started!")

# # Download dataset from Kaggle
# st.write("Downloading dataset from Kaggle...")
# try:
#     path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
#     st.write(f"Dataset downloaded to: {path}")
# except Exception as e:
#     st.error(f"Error downloading dataset: {e}")
#     st.stop()

# # Load YOLO model
# st.write("Loading YOLOv8 model...")
# try:
#     model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 model
#     st.success("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")
#     st.stop()

# # Find all image files in the dataset
# image_files = glob.glob(f"{path}/**/*.*", recursive=True)
# image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# st.write(f"Total images found: {len(image_files)}")
# if len(image_files) == 0:
#     st.error("No images found in the dataset.")
#     st.stop()

# # Streamlit app interface
# st.title("Animal Detection App")
# st.markdown("""
# Upload an image or use a random sample from the animal dataset to detect objects using YOLOv8.
# """)

# # Sidebar for settings
# st.sidebar.header("Settings")
# confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# # User chooses whether to use a random sample or upload an image
# random_sample = st.checkbox("Use Random Sample from Dataset", value=True)

# if random_sample:
#     # Select a random image from the dataset
#     image_path = random.choice(image_files)
#     st.write(f"Random sample selected: `{os.path.basename(image_path)}`")
# else:
#     # Let the user upload an image
#     uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(uploaded_file.read())
#             image_path = temp_file.name
#     else:
#         st.warning("Please upload an image or select random sampling to proceed.")
#         st.stop()

# # Load and process the image
# st.write("Processing the image...")
# image = cv2.imread(image_path)
# if image is None:
#     st.error("Failed to load the image!")
#     st.stop()

# # Convert image to RGB for display
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Display the original image
# st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

# # Run YOLO object detection
# st.write("Running YOLO object detection...")
# try:
#     results = model(image, conf=confidence_threshold)

#     # Annotate the image with bounding boxes and labels
#     annotated_image = results[0].plot()

#     # Display the annotated image
#     st.image(annotated_image, caption="Predicted Objects", use_column_width=True)

#     # List detected objects
#     detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
#     if detected_classes:
#         st.write(f"Objects Detected: {', '.join(set(detected_classes))}")
#     else:
#         st.write("No objects detected in the image.")
# except Exception as e:
#     st.error(f"Error during object detection: {e}")

# import streamlit as st
# import cv2
# from ultralytics import YOLO
# from PIL import Image
# import tempfile
# import numpy as np

# # Load YOLO model
# st.set_page_config(page_title="Object Detection App", layout="centered")
# st.title("🐾 Object Detection Application")
# st.markdown("""
# Welcome to the Object Detection Application! Upload an image to detect objects using the **YOLOv8** model.
# """)

# # Sidebar for settings
# st.sidebar.header("Settings")
# confidence_threshold = st.sidebar.slider(
#     "Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Adjust the confidence level for object detection."
# )

# # Load YOLO model
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8n.pt")  # Pretrained YOLOv8 model

# st.sidebar.write("Model: YOLOv8")
# model = load_model()

# # File uploader
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], help="Choose an image for object detection.")

# if uploaded_file:
#     # Save uploaded file to a temporary location
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_image_path = temp_file.name

#     # Read the image using OpenCV
#     image = cv2.imread(temp_image_path)
#     if image is None:
#         st.error("Failed to load the image. Please upload a valid image file.")
#         st.stop()

#     # Convert image to RGB for display
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

#     # Run object detection
#     st.write("Detecting objects...")
#     results = model(image, conf=confidence_threshold)

#     # Annotate the image with bounding boxes and labels
#     annotated_image = results[0].plot()

#     # Convert annotated image to display in Streamlit
#     annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

#     # Display annotated image
#     st.image(annotated_image_rgb, caption="Detected Objects", use_column_width=True)

#     # List detected objects
#     detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
#     if detected_classes:
#         st.success(f"Objects Detected: {', '.join(set(detected_classes))}")
#     else:
#         st.warning("No objects detected in the image.")
# else:
#     st.info("Please upload an image to start object detection.")


import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import numpy as np

# Configure Streamlit app
st.set_page_config(
    page_title="🐾 Animal Object Detection App",
    page_icon="🐾",
    layout="wide",
)

# App title and description
st.title("🐾 Animal Detection Application")
st.markdown("""
Welcome to the **Animal Detection Application**! 🦁  
Upload an image, and our powerful **YOLOv8** model will detect objects with bounding boxes.  
You can also adjust the confidence threshold in the sidebar for more accurate results.  
""")
st.markdown("---")

# Sidebar settings
st.sidebar.header("🔧 Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05, 
    help="Adjust the detection sensitivity."
)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

st.sidebar.write("📦 **Model Loaded**: YOLOv8")
model = load_model()

# File uploader for image input
uploaded_file = st.file_uploader(
    "📤 Upload an Image (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    help="Upload an image of an animal to detect objects."
)

# If an image is uploaded
if uploaded_file:
    st.markdown("### Uploaded Image")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_image_path = temp_file.name

    # Load image using OpenCV
    image = cv2.imread(temp_image_path)
    if image is None:
        st.error("Failed to load the image. Please upload a valid image file.")
        st.stop()

    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original Image", use_column_width=True)

    # Run YOLO object detection
    st.markdown("### Detecting Objects...")
    results = model(image, conf=confidence_threshold)

    # Annotate image with bounding boxes
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Extract detection results
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    confidences = [round(float(conf), 2) for conf in results[0].boxes.conf]
    boxes = [box.tolist() for box in results[0].boxes.xyxy]  # Bounding box coordinates

    # Display annotated image
    st.markdown("### Annotated Image")
    st.image(annotated_image_rgb, caption="Detected Objects", use_column_width=True)

    # Display detection results in a table
    if detected_classes:
        st.markdown("### Detection Results")
        df = pd.DataFrame({
            "Class": detected_classes,
            "Accuracy": confidences,
            "Bounding Box (X1, Y1, X2, Y2)": boxes,
        })
        st.table(df)

        # Add detection statistics to the sidebar
        st.sidebar.markdown("### 📊 Detection Statistics")
        st.sidebar.write(f"**Objects Detected:** {len(detected_classes)}")
        st.sidebar.write(f"**Unique Classes:** {len(set(detected_classes))}")
        st.sidebar.write("🎯 **Detected Classes:**")
        for obj in set(detected_classes):
            st.sidebar.write(f"- {obj}")
    else:
        st.warning("No objects were detected in the image. Try lowering the confidence threshold.")

else:
    st.info("Please upload an image to start detection.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [YOLOv8](https://ultralytics.com) and [Streamlit](https://streamlit.io).")

