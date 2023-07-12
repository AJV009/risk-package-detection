# Threat Detection and Unattended Baggage Detection
__YOLOv8s + OpenVINO + DeepSORT__

This is a demo of threat dtection and unattended baggage tracking using YOLOv4s and DeepSORT. The model is trained on [COCO](https://cocodataset.org/#home) dataset and the weights are converted to OpenVINO format. The model is then used to detect threats and unattended baggage in a video stream. The detections are then tracked using DeepSORT.

## Features
- Threat Detection - Detects `knife` and `scissor` classes.
- Unattended Baggage Detection - Detects `lagguage`, `backpack` and `handbag` classes and creates a relationship between the detections to the closest `person` to track the unattended baggage.
- Supports multiple persons and multiple unattended baggage tracking using __DeepSORT__.

## Current limitations in this demo:
- The bounding box logic isn't perfect and sometimes the bounding boxes just go outside the frame. (Which works but causes the center points to be outside the frame and thus the tracking fails)
- Since the model was trained with COCO its limited by the training data. (For example, it doesn't detect `gun` class because it wasn't in the training data or doesn't detect every kind of bag because all varieties of bags weren't in the training data)
- We are using the `s` version of YOLOv8 because we were having trouble with keeping consistency across the frames. Please note the `n` version works just fine also it was 2x fast than `s` but it might not be as accurate as the `s` version.

## How can it be improved:
- Training the model on a custom dataset with the specific classes we want to detect. (`person`, `knife`, `scissor`, `lagguage`, `backpack`, `handbag` and `gun`)
- Reusing the `n` version of YOLOv8 and fixing the consistency issue.
- Using a better bounding box logic to keep the bounding boxes inside the frame.
- Custom train the DeepSORT model to improve the tracking.

## How it works
1. The model is trained on COCO dataset using YOLOv8s. (We used the `s` instead of `n` because we were having trouble with keeping consistency across the frames)
2. The model is then converted to OpenVINO format using the `convert.ipynb` notebook.
3. The model is then used to detect threats and unattended baggage in a video stream:
    - Filter detections to only show the threats and persons and unattended baggage.
    - Create unique detections for each person and unattended baggage using __DeepSORT__.
    - Then create pairs for them, while also returning the alerts for threats.
    - Create a relationship between the unattended baggage and the closest person to track the unattended baggage.
    - Finally, draw the detections, keep track of the time period of the unattended baggage and draw the alerts.

## How to run
1. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Create the optimized model by running the notebook cells at `convert.ipynb` to convert the model to OpenVINO format. **(IMPORTANT)**
2. Run the cell at `main.ipynb` to run the demo. _(Make sure to change the video path to your supported video source)_
    
_Note: Demo won't work without creating the optimized model._

## Huge thanks to:
- To my sister for staying up late with me to test the demo and debug the detections. :heart:
- DeepSORT library: [theAIGuysCode/yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
- The binding logic for DeepSORT and YOLO: [DeepSORT-YOLOv4-TensorRT-OpenVINO](https://github.com/MatPiech/DeepSORT-YOLOv4-TensorRT-OpenVINO)
- Used as base repo for inference: [YOLOv8-OpenVINO-Optimised](https://github.com/AJV009/YOLOv8-OpenVINO-Optimised)

## Demo (CPU)

__Unattended Baggage Detection:__
![demo1](demoImages/demo1.png)
![demo2](demoImages/demo2.png)

__Thread Detection:__
![demo1](demoImages/demo3.png)

__More Demos:__
![demo1](demoImages/demo4.png)
![demo1](demoImages/demo5.png)
![demo1](demoImages/demo6.png)
