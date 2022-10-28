# limit the number of cpus used by high performance libraries
import os
import mysql.connector
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')

import argparse
import os
import numpy as np
import platform
import shutil
from datetime import datetime
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from stablization import make_thresh
from stablization import manage_greys
from stablization import set_buffer_memory


from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data = []
objects = {}
start_time = datetime.now().strftime('%H:%M:%S')
fps_count = []
avg_yolo_time = []
avg_sort_time = []
detection_intervels = []
total_info = {}
image_size = 0
movements = {}
img_queue = []


def trim_to_10(dict, list, index):
    if len(list) > index:
        list = list[-index:]
    for kry in dict:
        if len(dict[kry]) > index:
            dict[kry] = dict[kry][-index:]
    return dict, list





def detect(opt):
    fps1 = time_sync()
    global total_info, image_size, img_queue, movements, objects
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    save_path = 'Detected' + source.split('/')[-1]
    refer_vid = cv2.VideoCapture(source)
    while True:
        temp_ret, temp_frame = refer_vid.read()

        h,w,_ = temp_frame.shape
        break
    image_size = (w,h)
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            show_info = {}

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            labels = []
            if det is not None and len(det):
                for info in det:
                    key = class_names[int(info[-1])]
                    if key not in show_info.keys():
                        show_info[key] = []
                    show_info[key].append(float(info[-2]))
                for show_key in show_info:
                    if show_key not in total_info.keys():
                        total_info[show_key] = []
                    total_info[show_key].extend(show_info[show_key])
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:

                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # count
                        # count_obj(bboxes, w, h, id)
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        centeroid = annotator.box_label(bboxes, label, color=colors(c, True))
                        if id not in objects.keys():
                            objects[id] = centeroid
                        else:
                            objects[id] = centeroid

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                avg_yolo_time.append(t3 - t2)
                avg_sort_time.append(t5 - t4)
            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            show_vid = True
            if show_vid:
                global count
                color = (0, 255, 0)

                h, w, _ = im0.shape
                try:
                    for output in outputs:
                        x1, y1, x2, y2, id, cls = output
                        cenx = round((x2 - x1) / 2 + x1)
                        ceny = round((y2 - y1) / 2 + y1)
                        if id not in movements.keys():
                            movements[id] = []
                        movements[id].append((cenx, ceny))
                    img_queue.append((im0))
                    movements, img_queue = trim_to_10(movements, img_queue, 15)
                    for points in movements:
                        dist = 0
                        current_point = movements[points]
                        if len(current_point) > 10:
                            prev_pt = current_point[0]
                            next_pt = current_point[1]
                            for point in current_point[2:]:
                                dist = dist + np.linalg.norm(np.array(prev_pt) - np.array(next_pt))
                                prev_pt = next_pt
                                next_pt = point
                                # print(dist, f'for {points}')
                            if dist > 100:
                                for img in img_queue:
                                    vid_writer.write(img)
                                img_queue = []
                                break
                except:
                    pass

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3

                thickness = 3
                xh, yp, dep = im0.shape
                portion = yp // 10

                for heading in show_info:
                    avg = sum(show_info[heading]) / len(show_info[heading])

                    cv2.putText(im0, str(heading), (0, portion // 2,), font,
                                fontScale, color, thickness, cv2.LINE_AA)
                    cv2.putText(im0, str(avg)[:4], (0, portion), font,
                                fontScale, color, thickness, cv2.LINE_AA)
                    portion += portion
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    vid_writer.release()
                    raise StopIteration

    fps2 = time_sync()
    fps_count.append(fps2 - fps1)
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    vid_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str,
                        default='/home/icrl/PycharmProjects/Saad/yolov5 tracker/DeepSORT_YOLOv5_Pytorch-master/yolov5s.pt',
                        help='model.pt path(s)')
    # Database Credentials
    parser.add_argument('--user', type=str, )
    parser.add_argument('--password', type=str, )
    parser.add_argument('--host', type=str,default='localhost' )
    parser.add_argument('--database', type=str, )
    parser.add_argument('--table', type=str, )
    parser.add_argument('--VM_ID', type=str, )
    args = parser.parse_args()
    # Database Connection
    cnx = mysql.connector.connect(user=args.user,
                                  password=args.password,
                                  host=args.host,
                                  database=args.database)
    print(args.VM_ID)
    cursor = cnx.cursor(buffered=True)
    print('cursor established')

    fetch_query = f'SELECT * FROM {args.database}.{args.table} WHERE video_vm_id=%s'

    cursor.execute(fetch_query, tuple(args.VM_ID))
    print('quesry done')

    fetched_data = [(i[0],i[4]) for i in cursor]
    print(fetched_data)
    for vid_id, source in fetched_data:
        print(vid_id, source)
    # main processing
        parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
        parser.add_argument('--source', default=source, type=str, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='inference/output', help='output folWder')  # output folder
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--show-vid', default=True, action='store_false', help='display tracking video results')
        parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
        parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
        # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--evaluate', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
        parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

        with torch.no_grad():
            count = 0
            data = []
            objects = {}
            start_time = datetime.now().strftime('%H:%M:%S')
            fps_count = []
            avg_yolo_time = []
            avg_sort_time = []
            detection_intervels = []
            total_info = {}
            image_size = 0
            movements = {}
            img_queue = []
            detect(opt)
        # stats
        end_time = datetime.now().strftime('%H:%M:%S')
        video_yolo_time = sum(avg_yolo_time) / len(avg_yolo_time)
        video_sort_time = sum(avg_sort_time) / len(avg_sort_time)
        # get all avgs
        for info in total_info:
            total_info[info] = sum(total_info[info]) / len(total_info[info])
        total_info = {k: v for k, v in sorted(total_info.items(), key=lambda item: item[1])}
        objects_detected = list(total_info.keys())
        image_size = str(image_size[0]) + ', ' + str(image_size[1])
        conf_data = [str(val) for val in total_info.values()]
        print(start_time, end_time, image_size, str(video_sort_time), str(video_yolo_time), list(total_info.keys())[::-1], ', '.join(conf_data[::-1]),
              str(sum(fps_count) / len(fps_count)))
        insert_query = 'UPDATE testDB.video SET stats_start_time = %s, stats_end_time = %s, stats_image_size = %s, stats_objects_count = %s, stats_objects_detection = %s, stats_fps = %s,  stats_yolo_sort_time =  %s, stats_detection_time = %s WHERE (video_id = %s);'
        insert_vals = (start_time, end_time, image_size, ', '.join(list(total_info.keys())[::-1]), ', '.join(conf_data[::-1]), sum(fps_count) / len(fps_count), video_sort_time, video_yolo_time, vid_id)
        cursor.execute(insert_query, insert_vals)
        cnx.commit()
    cursor.close()
    cnx.close()