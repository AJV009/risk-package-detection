# Py imports
import os
import threading
import time
import urllib.parse
from typing import Dict, Tuple
from os import PathLike
from pathlib import Path
from datetime import datetime

# Third party imports
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from scipy.optimize import linear_sum_assignment
from openvino.runtime import Model
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.plotting import colors as ucolors

# Local imports
from deepSORT import preprocessing
from deepSORT.detection import Detection

# write a simple function that take any input and saves it to a file like log.txt
def log_output(output):
    with open('log.txt', 'a') as f:
        f.write(str(output)+'\n')

class VideoPlayer:
    """
    Custom video player to fulfill FPS requirements. You can set target FPS and output size,
    flip the video horizontally or skip first N frames.

    :param source: Video source. It could be either camera device or video file.
    :param size: Output frame size.
    :param flip: Flip source horizontally.
    :param fps: Target FPS.
    :param skip_first_frames: Skip first N frames.
    """

    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0):
        import cv2

        self.cv2 = cv2  # This is done to access the package in class methods
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(
                f"Cannot open {'camera' if isinstance(source, int) else ''} {source}"
            )
        # skip first N frames
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            # AREA better for shrinking, LINEAR better for enlarging
            self.__interpolation = (
                cv2.INTER_AREA
                if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                else cv2.INTER_LINEAR
            )
        # first frame
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

    """
    Start playing.
    """

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    """
    Stop playing and release resources.
    """

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

        self.__frame = None

    """
    Get current frame.
    """

    def next(self):
        import cv2

        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = self.cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = self.cv2.flip(frame, 1)
        return frame

def plot_one_box(box:np.ndarray, img:np.ndarray, color:Tuple[int, int, int] = None, mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
        img (no.ndarray): input image
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img

def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
    """
    Resize image and padding for detection. Takes image as input, 
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints
    
    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size
    
    
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def postprocess(
    pred_boxes:np.ndarray, 
    input_hw:Tuple[int, int], 
    orig_img:np.ndarray, 
    min_conf_threshold:float = 0.25, 
    nms_iou_threshold:float = 0.7, 
    agnosting_nms:bool = False, 
    max_detections:int = 300,
    pred_masks:np.ndarray = None,
    retina_mask:bool = False
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
        pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
        retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=80,
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results

def clip_bbox(bbox, image_shape):
    """
    Clip the bounding box coordinates to ensure they fall within the image boundaries.
    
    Parameters:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        tuple: Clipped bounding box coordinates (x1, y1, x2, y2).
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    return (x1, y1, x2, y2)

def process_results(results:Dict, source_image:np.ndarray, label_map:Dict, deepsort_config:Dict):
    """
    Helper function for drawing bounding boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:
        
    """
    boxes = results["det"]
    h, w = source_image.shape[:2]
    objects = []

    scores = []
    classes = []
    names = []
    bboxes = []

    track_classes = deepsort_config["track_classes"]
    threat_classes = deepsort_config["threat_classes"]

    alert = []

    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        xyxy = clip_bbox(xyxy, source_image.shape)
        if label_map[int(lbl)] in track_classes and conf > 0.45:
            objects.append({"object": idx, "xyxy": xyxy, "label": label_map[int(lbl)], "print_label": f'{label_map[int(lbl)]} {conf:.2f}', "confidence": conf})

            bboxes.append(xyxy)
            scores.append(f'{conf:.2f}')
            classes.append(int(lbl))
            names.append(label_map[int(lbl)])
        elif label_map[int(lbl)] in threat_classes and conf > 0.45:
            # single box plotter
            label = f'{label_map[int(lbl)]} {conf:.2f}'
            source_image = plot_one_box(xyxy, source_image, label=label, color=ucolors(int(lbl)), line_thickness=1)
            alert.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Threat {label_map[int(lbl)]} detected")
    
    # tracker feed
    encoder = deepsort_config["encoder"]
    features = encoder(source_image, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

    # Init color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # Run non-maxima suppression.
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    nms_max_overlap = deepsort_config["nms_max_overlap"]
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker = deepsort_config["tracker"]
    tracker.predict()
    tracker.update(detections)

    # store confident tracks for further processing
    confident_tracks = []

    # update tracks (use plot_one_box)
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        color = colors[int(track.track_id) % len(colors)]
        track_data = {
            'bbox': track.to_tlbr(),
            'color': tuple([i * 255 for i in color]),
            'label': f'{track.get_class()} {track.track_id}',
        }
        confident_tracks.append(track_data)
        source_image = plot_one_box(track_data["bbox"], source_image, label=track_data["label"], color=track_data["color"], line_thickness=1)

    return source_image, confident_tracks, alert

def get_center(bbox):
    # Get the center point of the bounding box
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance(center1, center2):
    # Calculate Euclidean distance between two points
    return np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

def track_risk(frame, deepsort_config, relation_history, confident_tracks):
    alert = []

    people_tracks = [track for track in confident_tracks if track['label'].startswith('person')]
    object_tracks = [track for track in confident_tracks if track['label'].startswith(tuple(deepsort_config['baggage_classes']))]

    num_people = len(people_tracks)
    num_objects = len(object_tracks)
    cost_matrix = np.full((num_people, num_objects), np.inf)

    current_person_labels = [track['label'] for track in people_tracks]

    for i, person in enumerate(people_tracks):
        for j, obj in enumerate(object_tracks):
            center_person = get_center(person['bbox'])
            center_object = get_center(obj['bbox'])
            temp_pair = (person['label'], obj['label'])
            if temp_pair in relation_history:
                cost_matrix[i][j] = get_distance(center_person, center_object) - deepsort_config['history_weight']
            else:
                cost_matrix[i][j] = get_distance(center_person, center_object)

    alert_config = deepsort_config['alert_config']['unattended_bag']

    if num_people > 0 and num_objects > 0: 
        person_indices, object_indices = linear_sum_assignment(cost_matrix)
        for person_index, object_index in zip(person_indices, object_indices):
            center_person = get_center(people_tracks[person_index]['bbox'])
            center_object = get_center(object_tracks[object_index]['bbox'])
            dist = get_distance(center_person, center_object)
            temp_pair = (people_tracks[person_index]['label'], object_tracks[object_index]['label'])
            if temp_pair not in relation_history:
                relation_history[temp_pair] = {
                    'distances': [dist],
                    'timestamps': [time.time()]
                }
            elif dist < alert_config['pixel_distance_percent'] * frame.shape[1]:
                if 'unattended_since' in relation_history[temp_pair]:
                    alert.append(f'{datetime.now()}: Alert resolved: {temp_pair[0]}-{temp_pair[1]}')
                    del relation_history[temp_pair]['unattended_since']
                relation_history[temp_pair]['distances'].append(dist)
                relation_history[temp_pair]['timestamps'].append(time.time())
            else:
                last_seen = max(relation_history[temp_pair]['timestamps'])
                if time.time() - last_seen > alert_config['alert_duration']:
                    if 'unattended_since' not in relation_history[temp_pair]:
                        alert.append(f'{datetime.now()}: Unattended bag detected: {temp_pair[0]}-{temp_pair[1]}')
                        relation_history[temp_pair]['unattended_since'] = time.time()
            cv2.line(frame, center_person, center_object, people_tracks[person_index]['color'], 2)


    for temp_pair in list(relation_history.keys()):
        person_label, object_label = temp_pair
        if person_label not in current_person_labels:
            last_seen = max(relation_history[temp_pair]['timestamps'])
            if time.time() - last_seen > alert_config['alert_duration']: 
                if 'unattended_since' not in relation_history[temp_pair]:
                    alert.append(f'{datetime.now()}: Unattended bag detected: {person_label}-{object_label}')
                    relation_history[temp_pair]['unattended_since'] = time.time()
                elif time.time() - relation_history[temp_pair]['unattended_since'] > alert_config['grace_period']:
                    alert.append(f'{datetime.now()}: Bag exceeded grace period: {person_label}-{object_label}')

    grace_period = alert_config['grace_period']
    for temp_pair in list(relation_history.keys()):
        if 'unattended_since' in relation_history[temp_pair] and time.time() - relation_history[temp_pair]['unattended_since'] > grace_period:
            del relation_history[temp_pair]

    return frame, relation_history, alert

def detect_without_preprocess(image:np.ndarray, model:Model):
    """
    OpenVINO YOLOv8 model with integrated preprocessing inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
    output_layer = model.output(0)
    img = letterbox(image)[0]
    input_tensor = np.expand_dims(img, 0)
    input_hw = img.shape[:2]
    result = model(input_tensor)[output_layer]
    detections = postprocess(result, input_hw, image)
    return detections

def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
    silent: bool = False,
    timeout: int = 10,
) -> PathLike:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    from tqdm.notebook import tqdm_notebook
    import requests

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / Path(filename)

    try:
        response = requests.get(url=url, 
                                headers={"User-agent": "Mozilla/5.0"}, 
                                stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
                "Connection timed out. If you access the internet through a proxy server, please "
                "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist, or if it exists with an incorrect file size
    filesize = int(response.headers.get("Content-length", 0))
    if not filename.exists() or (os.stat(filename).st_size != filesize):

        with tqdm_notebook(
            total=filesize,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(filename),
            disable=not show_progress,
        ) as progress_bar:

            with open(filename, "wb") as file_object:
                for chunk in response.iter_content(chunk_size):
                    file_object.write(chunk)
                    progress_bar.update(len(chunk))
                    progress_bar.refresh()
    else:
        if not silent:
            print(f"'{filename}' already exists.")

    response.close()

    return filename.resolve()

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

class ImageEncoderTF(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoderTF(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)
    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    # Preserve aspect ratio, resize the largest dimension to match the model's expected size.
    scale = min(patch_shape[0] / image.shape[0], patch_shape[1] / image.shape[1])
    image = cv2.resize(image, None, fx=scale, fy=scale)
    # Pad the image to match the model's expected size.
    pad_y = (patch_shape[0] - image.shape[0]) // 2
    pad_x = (patch_shape[1] - image.shape[1]) // 2
    image = np.pad(image, ((pad_y, patch_shape[0] - image.shape[0] - pad_y), (pad_x, patch_shape[1] - image.shape[1] - pad_x), (0, 0)), mode='constant')

    return image

    # if patch_shape is not None:
    #     # correct aspect ratio to patch shape
    #     target_aspect = float(patch_shape[1]) / patch_shape[0]
    #     new_width = target_aspect * bbox[3]
    #     bbox[0] -= (new_width - bbox[2]) / 2
    #     bbox[2] = new_width
    # else:
    #     patch_shape = np.round(bbox[2:]).astype(np.int)

    # # convert to top left, bottom right
    # bbox[2:] += bbox[:2]
    # bbox = bbox.astype(np.int)

    # # clip at image boundaries
    # bbox[:2] = np.maximum(0, bbox[:2])
    # bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    # if np.any(bbox[:2] >= bbox[2:]):
    #     return None
    # sx, sy, ex, ey = bbox
    # image = image[sy:ey, sx:ex]
    # image = cv2.resize(image, tuple(patch_shape[::-1]))
    # return image

