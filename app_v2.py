import os

import random
import numpy as np

import torch
# import torch.nn.functional as F
# from torchvision.transforms import Compose

import cv2
import PIL
from PIL import Image

import gradio as gr

from backbones.basenet import MobileNet_GDConv
from backbones.mobilefacenet import MobileFaceNet
from backbones.pfld_compressed import PFLDInference
from detectors.Retinaface import RetinaFaceNet as RetinaFace
from detectors.FaceBoxes import FaceBoxes
from detectors.MTCNN import detect_faces
from common.utils import BBox, drawLandmark, drawLandmark_multiple
from utils.align_trans import get_reference_facial_points, warp_and_crop_face


# Global Variables
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

CROP_SIZE = 112
SCALE = CROP_SIZE / 112.

FACE_CONFIDENCE = 0.9
FACE_REFERENCE = get_reference_facial_points(default_square=True) * SCALE

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'


# Model Config
face_detectors = ["FaceBoxes", "RetinaFace", "MTCNN"]
face_landmarkers = ["MobileNet", "MobileFaceNet", "PFLD"]


def load_model(backbone: str = 'MobileFaceNet'):

    backbone = backbone.lower()
    print(f'\n\nLoading {backbone} as backbone ...')

    global Model

    if backbone == 'mobilenet':
        Model = MobileNet_GDConv(136)
        Model = torch.nn.DataParallel(Model)
        checkpoint_fn = 'mobilenet_224_model_best_gdconv_external'

    elif backbone == 'pfld':
        Model = PFLDInference()
        checkpoint_fn = 'pfld_model_best'
    
    elif backbone == 'mobilefacenet':
        Model = MobileFaceNet([112, 112], 136)
        checkpoint_fn = 'mobilefacenet_model_best'
    
    else:
        print('Error: not suppored backbone')

    checkpoint_path = f'checkpoints/{checkpoint_fn}.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    Model.load_state_dict(checkpoint['state_dict'])
    Model = Model.eval()


def run_detection(image_path: str, detector: str = 'RetinaFace'):

    print(f'\nRunning face-detection for {image_path} ...')

    detector = detector.lower()

    image_fn = os.path.splitext(
                os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Face detection
    if detector == 'mtcnn':
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # image = Image.open(image_path)
        faces, landmarks = detect_faces(image)

    else:
        if detector == 'faceboxes':
            face_detector = FaceBoxes()
        elif detector == 'retinaface':
            face_detector = RetinaFace()    
        else:
            raise ValueError(f"{detector} is not supported!")
        faces = face_detector(img)

    ratio = 0
    if len(faces) == 0:
        print('NO face is detected!')

    faces_list = []
    for k, face in enumerate(faces):
        if face[4] < FACE_CONFIDENCE: 
            # remove low confidence detection
            continue

        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)

        cropped = img[  new_bbox.top  : new_bbox.bottom, 
                        new_bbox.left : new_bbox.right  ]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)

        if cropped.shape[0] <= 0 or cropped.shape[1] <= 0:
            continue

        # save cropped face
        face_path = os.path.join('results', f"{image_fn}_{k:02d}.png")
        faces_list.append(face_path)
        cv2.imwrite(face_path, cropped)

    return faces_list
            

def focus_5_points(landmark: np.ndarray, image: np.ndarray = None):

    # Crop and aligned the face
    lefteye_x = 0
    lefteye_y = 0
    for i in range(36, 42):
        lefteye_x += landmark[i][0]
        lefteye_y += landmark[i][1]
    lefteye_x = lefteye_x / 6
    lefteye_y = lefteye_y / 6
    lefteye = [lefteye_x, lefteye_y]

    righteye_x = 0
    righteye_y = 0
    for i in range(42, 48):
        righteye_x += landmark[i][0]
        righteye_y += landmark[i][1]
    righteye_x = righteye_x / 6
    righteye_y = righteye_y / 6
    righteye = [righteye_x, righteye_y]  

    nose = landmark[33]
    leftmouth = landmark[48]
    rightmouth = landmark[54]
    focus_points = [righteye, lefteye, nose, rightmouth, leftmouth]
    
    if image is None:
        return focus_points

    warped_face = warp_and_crop_face(np.array(image), focus_points, FACE_REFERENCE, crop_size=(CROP_SIZE, CROP_SIZE))
    warped_face = Image.fromarray(warped_face)

    return focus_points, warped_face


def run_landmark(image_path: str, 
                 landmarker: str = 'MobileFaceNet', 
                 focus_mode: str = 'false'):

    print(f'\nRunning face-landmark for {image_path} ...')

    global Model

    landmarker = landmarker.lower()
    if landmarker == 'mobilenet':
        out_size = 224
    else:
        out_size = 112

    img = cv2.imread(image_path)
    H, W, _ = img.shape
    bbox = BBox([0, W, 0, H])

    cropped_face = cv2.resize(img, (out_size, out_size))
    # if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
    #     continue

    test_face = cropped_face.copy()
    test_face = test_face / 255.
    if landmarker == 'mobilenet':
        test_face = (test_face - mean) / std
    test_face = test_face.transpose((2, 0, 1))
    test_face = test_face.reshape((1,) + test_face.shape)

    face_arr = torch.from_numpy(test_face).float()
    face_arr = torch.autograd.Variable(face_arr)

    if landmarker == 'mobilefacenet':
        landmark = Model(face_arr)[0].cpu().data.numpy()
    else:
        landmark = Model(face_arr).cpu().data.numpy()

    landmark = landmark.reshape(-1,2)
    landmark = bbox.reprojectLandmark(landmark)
    if str(focus_mode).lower() in ['true', 't', 'y', '1']:
        landmark = focus_5_points(landmark)

    img = drawLandmark_multiple(img, bbox, landmark)
    
    # save the landmark detections
    output_fn = os.path.splitext(
                os.path.basename(image_path))[0] + '_marked.png'
    output_path = os.path.join('results', output_fn)
    cv2.imwrite(output_path, img)
    return output_path


# Define styles
css = """
.gradio-container {width: 85% !important}
"""

# Define texts
title = r"""
<h1 align="center">Face Reading: Facial Landmark Detection and Physiognomy</h1>
"""

description = r"""
<b>Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b> InstantID </b></a>.<br>
"""

tips = r"""
### Usage tips of Face Detection
1. If you're not satisfied, ..."
2. If you feel that ...
"""


def load_ui():

    with gr.Blocks(css=css, analytics_enabled=False) as gui:
        
        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        # Body
        with gr.Tab(label="Face Detection"):
            with gr.Row():

                with gr.Column():
                    img_det_in = gr.Image(label="Upload a photo of face(s)", type="filepath")
                    detector = gr.Dropdown(label="Face Detector", choices=face_detectors, value="RetinaFace", interactive=True)
                
                with gr.Column(scale=1):
                    img_det_out = gr.Gallery(label="Detected face(s)", allow_preview=True, preview=True)
                    detection = gr.Button(value="Detect face(s)", variant="primary")
                    detection_tips = gr.Markdown(label="Detection Tips", value=tips, visible=True)

        with gr.Tab(label="Face Landmarker"):
            with gr.Row():
                
                with gr.Column():
                    img_lm_in = gr.Image(label="Upload a photo of single full face", type="filepath")
                    
                    with gr.Accordion(open=False, label="Advanced Options"):
                        landmarker = gr.Dropdown(label="Face Landmarker", choices=face_landmarkers, value="MobileFaceNet", interactive=True)
                        focus_mode = gr.Dropdown(label="Focus Mode", choices=['true', 'false'], value='false', interactive=True)
                        reloader = gr.Button(value="Re-Load model", variant="primary")

                with gr.Column(scale=1):
                    img_lm_out = gr.Image(label="Face Landmarks", type="pil")
                    facemarker = gr.Button(value="Highlight Face Landmarks", variant="primary")
                    facemarker_tips = gr.Markdown(label="Landmarking Tips", value=tips, visible=True)

        # Callback
        reloader.click(fn=load_model, inputs=landmarker)
        detection.click(fn=run_detection, inputs=[img_det_in, detector], outputs=img_det_out)
        facemarker.click(fn=run_landmark, inputs=[img_lm_in, landmarker, focus_mode], outputs=img_lm_out)

    return gui


if __name__ == "__main__":
    
    global Model
    load_model('MobileFaceNet')

    demo = load_ui()
    demo.queue(max_size=10, api_open=True)
    demo.launch(debug=True)

