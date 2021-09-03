from __future__ import division
from models import Darknet
from utils.utils import load_classes, non_max_suppression_output
import argparse
import time
import torch
import numpy as np
from torch.autograd import Variable
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_35.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--frame_size", type=int, default=832, help="size of each image dimension")  # 416
    parser.add_argument("--image", type=str, help="image name (with archive extension)")
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # checking for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.frame_size).to(device)

    # loading weights
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)  # Load weights
    else:
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device(device)))  # Load checkpoints

    # Set in evaluation mode
    model.eval()

    # Extracts class labels from file
    classes = load_classes(opt.class_path)

    # checking for GPU for Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Read image
    frame = t_frame = cv2.imread(opt.image)
    # Image feed dimensions
    v_height, v_width = frame.shape[:2]

    # For a black image
    x = y = v_height if v_height > v_width else v_width

    # Putting original image into black image
    start_new_i_height = int((y - v_height) / 2)
    start_new_i_width = int((x - v_width) / 2)

    # For accommodate results in original frame
    mul_constant = x / opt.frame_size

    # for text in output
    t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    start = time.time()

    # Black image
    frame = np.zeros((x, y, 3), np.uint8)

    frame[start_new_i_height: (start_new_i_height + v_height),
    start_new_i_width: (start_new_i_width + v_width)] = t_frame

    # resizing to [416x 416]
    frame = cv2.resize(frame, (opt.frame_size, opt.frame_size))
    # [BGR -> RGB]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # [[0...255] -> [0...1]]
    frame = np.asarray(frame) / 255
    # [[3, 416, 416] -> [416, 416, 3]]
    frame = np.transpose(frame, [2, 0, 1])
    # [[416, 416, 3] => [416, 416, 3, 1]]
    frame = np.expand_dims(frame, axis=0)
    # [np_array -> tensor]
    frame = torch.Tensor(frame)

    # [tensor -> variable]
    frame = Variable(frame.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(frame)
    detections = non_max_suppression_output(detections, opt.conf_thres, opt.nms_thres)

    # For each detection in detections
    with_mask = without_mask = 0
    detection = detections[0]
    if detection is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            # Accommodate bounding box in original frame
            x1 = int(x1 * mul_constant - start_new_i_width)
            y1 = int(y1 * mul_constant - start_new_i_height)
            x2 = int(x2 * mul_constant - start_new_i_width)
            y2 = int(y2 * mul_constant - start_new_i_height)

            print(cls_conf, cls_pred)

            # Bounding box making and setting Bounding box title
            if int(cls_pred) == 0 or cls_conf < 0.998 and int(cls_pred) == 1:
                # WITH_MASK
                with_mask += 1
                cv2.rectangle(t_frame, (x1, y1), (x2, y2), (0, 255, 0), 2).copy()
            else:
                # WITHOUT_MASK
                without_mask += 1
                cv2.rectangle(t_frame, (x1, y1), (x2, y2), (0, 0, 255), 2).copy()

    print("Numero de pessoas:", with_mask+without_mask)
    print("Pessoas com mascara:", with_mask)
    print("Pessoas sem mascara:", without_mask)
    cv2.imwrite('output.jpg', t_frame)
    print("Escrito em output.jpg")
