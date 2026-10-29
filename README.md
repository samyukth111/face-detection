# Face Finder

A small Streamlit app that detects faces in uploaded photos with OpenCV's YuNet
neural-network detector. It draws a confidence-labeled box around each
detection and lets you download the annotated image.

## Run on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, then upload a JPG, PNG, or WebP photo.
Use the sidebar to adjust detection sensitivity and the minimum face size.

Images are processed in memory by the app and are not saved. The displayed
number is the model's count of detected faces, not a guarantee of the true
number in every image; small, obscured, profile, or poorly lit faces can be
missed, and false detections are possible. Raise the face-confidence threshold
to reduce likely false positives.

The YuNet model and its MIT license are included in `models/`. Model credit:
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

## Run tests

```powershell
python -m unittest discover -s tests -v
```
