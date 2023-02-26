import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from io import BytesIO
import requests
from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT

Parameters = st.sidebar.title('Parameters')
# Define default threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Create a slider for the confidence threshold in the sidebar
confidence_threshold = st.sidebar.slider(
    "Set confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.05
)

# Use the selected threshold to update the DEFAULT_CONFIDENCE_THRESHOLD variable
DEFAULT_CONFIDENCE_THRESHOLD = confidence_threshold

# Use the updated DEFAULT_CONFIDENCE_THRESHOLD variable in your code
add_slider = (DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT)

@st.cache_data
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


@st.cache_data
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels


st.sidebar.title('About the App developer')
st.sidebar.write('Adeleke Olorunnisola, a mechanical engineer with an inherent passion for the investigation' 
                 '\n and assembly of cutting-edge intelligent systems and robotics')


st.header("Let's perform some object detection")

with st.container():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="fileUploader", 
                                       accept_multiple_files=False)
    if img_file_buffer:
        image = np.array(Image.open(img_file_buffer))


if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

elif url := st.text_input("Enter the URL of an image"):
    # Fetch the image from the URL using requests
    response = requests.get(url)
    image = np.array(Image.open(BytesIO(response.content)))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))


detections = process_image(image)
image, labels = annotate_image(image, detections, confidence_threshold)

st.image(
    image, caption=f"Processed image", use_column_width=True,
)

st.write(labels)
