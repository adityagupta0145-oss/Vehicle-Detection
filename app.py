import streamlit as st
import cv2
import tempfile
import time
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="🚗 Vehicle Detection System",
    page_icon="🚘",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 1rem;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 AI Vehicle Detection Dashboard")
st.markdown("### Upload a video and detect vehicles in real-time")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    scale_factor = st.slider(
        "Scale Factor",
        min_value=1.01,
        max_value=1.5,
        value=1.1,
        step=0.01
    )

    min_neighbors = st.slider(
        "Min Neighbors",
        min_value=1,
        max_value=10,
        value=3
    )

    st.markdown("---")
    st.info("Upload a video file to start detection.")

# File Upload
uploaded_file = st.file_uploader(
    "📤 Upload Video",
    type=["mp4", "avi", "mov"]
)

# Metrics Row
col1, col2, col3 = st.columns(3)

with col1:
    vehicle_metric = st.empty()

with col2:
    frame_metric = st.empty()

with col3:
    status_metric = st.empty()

# Detection Area
video_placeholder = st.empty()

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    car_cascade = cv2.CascadeClassifier(
    r"D:\Python 311\Vehicle Detection\cars.xml"
)

    frame_count = 0
    total_vehicles = 0

    start_btn = st.button("▶ Start Detection")

    if start_btn:

        progress_bar = st.progress(0)

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            cars = car_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors
            )

            total_vehicles += len(cars)

            for (x, y, w, h) in cars:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            vehicle_metric.metric(
                "🚗 Vehicles Detected",
                len(cars)
            )

            frame_metric.metric(
                "🎞 Frames Processed",
                frame_count
            )

            status_metric.metric(
                "📡 Status",
                "Running"
            )

            progress_bar.progress(
                min(frame_count / 500, 1.0)
            )

        cap.release()

        st.success("✅ Detection Completed")

        st.balloons()

        st.subheader("📊 Detection Summary")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Total Frames",
                frame_count
            )

        with c2:
            st.metric(
                "Total Vehicle Detections",
                total_vehicles
            )