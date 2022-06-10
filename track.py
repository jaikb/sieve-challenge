# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import json

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    source, yolo_model, deep_sort_model, imgsz, half, update = opt.source, opt.yolo_model, opt.deep_sort_model, opt.imgsz, opt.half, opt.update
    
    prefix, format = source.split(".")
    SAVE_PATH = prefix + '_results.mp4'
    RESULT_JSON = 'results.json'

    unc_colors = (211, 160, 86)
    duke_colors = (255, 255, 255)
    referee_colors = (0, 0, 0)

    device = select_device(opt.device)
    half &= device.type != 'cpu'

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    vid_path, vid_writer = None, None

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    with open(RESULT_JSON, 'w+') as f:
        result_dict = {'results' : []}
        json.dump(result_dict, f)

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path

            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                result_dict = {'frame_number' : frame_idx, 'objects' : []}
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):

                        bboxes = output[0:4]
                        id = output[4]

                        xmin, ymin, xmax, ymax = bboxes.astype(int)

                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]

                        img = im0s[ymin:ymax, xmin:xmax]
                        
                        if abs(ymin-ymax) < 120:
                            label = 'other'
                        else:
                            label = find_label(img)
                        
                        json_label = ""
                        
                        if label == 'duke':
                            annotator.box_label(bboxes, label, color=duke_colors, txt_color=(0, 0, 0))
                            json_label = "player_white_jersey"
                        elif label == 'unc':
                            annotator.box_label(bboxes, label, color=unc_colors)
                            json_label = "player_light_blue_jersey"
                        elif label == 'referee':
                            annotator.box_label(bboxes, label, color=referee_colors)
                            json_label = "referee"
                        elif label == 'other':
                            annotator.box_label(bboxes, label)
                            json_label = "other"


                        result_dict['objects'].append(
                            {"object_id" : id, "box_coordinates": (bbox_left, bbox_top, bbox_w, bbox_h), "person_type" : json_label}
                        )

                        
                with open(RESULT_JSON, 'r+') as f:
                    file_data = json.load(f)
                    file_data['results'].append(result_dict)
                    f.seek(0)
                    json.dump(file_data, f, indent=4)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            im0 = annotator.result()
            # Save results (image with detections)
            if vid_path[i] != SAVE_PATH:
                vid_path[i] = SAVE_PATH
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()
                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else: 
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                SAVE_PATH = str(Path(SAVE_PATH).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    LOGGER.info(f"Results saved to {source}_results.mp4 and results.json")

    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


# Use masking to get player jersey colors
def find_label(image):
    team_list=['referee','unc','duke']
    
    boundaries = [
        ([0, 0, 0], [50, 50, 50]), #referee
        ([43, 31, 4], [211, 160, 86]), #unc
        ([199, 189, 155],[255, 255, 255]) #duke
    ]
    
    mask_dict = {}
    
    for team, boundary in zip(team_list, boundaries):
        mask = cv2.inRange(image,  np.array(boundary[0]),  np.array(boundary[1]))
        mask_dict[team] = mask
    
    output_dict = {}
    
    for team, mask in mask_dict.items():
        output_dict[team] = cv2.bitwise_and(image, image, mask = mask)
    
    ratio_dict = {}
    for team, output in output_dict.items():
        total_pix = image.any(axis=-1).sum()
        color_pix = output.any(axis=-1).sum()
        ratio_dict[team] = color_pix / total_pix
        
    return max(ratio_dict, key=ratio_dict.get)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
