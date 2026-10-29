from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from face_detector import detect_faces, load_face_detector

st.set_page_config(
    page_title="Face Finder",
    page_icon=":material/face:",
    layout="wide",
)

st.title("Face Finder")
st.markdown("Count faces in a photo and download an annotated copy.")
st.caption(
    "Images are processed in memory by this app and are not saved. "
    "Counts are model detections, not a guarantee; clear, front-facing photos "
    "give the most reliable results."
)

with st.sidebar:
    st.header("Detection settings")
    confidence_threshold = st.slider(
        "Face confidence threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.85,
        step=0.05,
        help="Higher values reduce false positives but may miss faces.",
    )
    min_face_size = st.slider(
        "Minimum face size (pixels)",
        min_value=20,
        max_value=200,
        value=30,
        step=5,
    )

uploaded_image = st.file_uploader(
    "Choose a photo",
    type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=10,
    help="Upload a JPG, PNG, or WebP image up to 10 MB.",
)

if uploaded_image is None:
    st.info("Upload a photo to start detecting faces.")
else:
    try:
        with Image.open(uploaded_image) as source:
            if source.width * source.height > 24_000_000:
                st.error("This image is too large to process. Choose an image under 24 megapixels.")
                st.stop()
            original = source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("This file could not be opened as an image. Try another photo.")
        st.stop()

    detector = load_face_detector()
    annotated, faces = detect_faces(
        original,
        detector,
        min_face_size=min_face_size,
        confidence_threshold=confidence_threshold,
    )

    with st.container(border=True):
        metric_columns = st.columns(3)
        metric_columns[0].metric("Faces detected", len(faces))
        metric_columns[1].metric("Image dimensions", f"{original.width} × {original.height}")
        metric_columns[2].metric("Detection method", "Frontal face")

    original_column, result_column = st.columns(2)
    with original_column:
        st.subheader("Original")
        st.image(original, alt="Uploaded photo before face detection", width="stretch")
    with result_column:
        st.subheader("Detected faces")
        st.image(annotated, alt="Uploaded photo with detected faces outlined", width="stretch")

    if faces:
        st.success(f"Detected {len(faces)} face{'s' if len(faces) != 1 else ''}.")
    else:
        st.warning("No faces found. Try a clearer, front-facing photo or adjust the settings.")

    download = BytesIO()
    annotated.save(download, format="PNG")
    output_name = f"{Path(uploaded_image.name).stem}_faces.png"
    st.download_button(
        "Download annotated image",
        data=download.getvalue(),
        file_name=output_name,
        mime="image/png",
        icon=":material/download:",
    )
