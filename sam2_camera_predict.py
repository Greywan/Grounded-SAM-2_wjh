import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from tqdm import tqdm

from yolo.yolo_function import yolo_preprocess
from utils.calculate_time import splitting_time_eachnum, TimeCalculator
from loguru import logger
from datetime import datetime
from utils.vis import draw_boxs_specifycolor,draw_mask_specifycolor
from mask_propocess import MaskProcessor

def main(video_dir, output_dir, step, logger):
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2_camera_predictor

    sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"

    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    video_dir = "./notebooks/videos/aquarium/aquarium"
    # cap = cv2.VideoCapture("./notebooks/videos/aquarium/aquarium.mp4")

    frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img_paths = [os.path.join(video_dir, frame_name) for frame_name in frame_names]

    if_init = False

    frame_list = []
    for frame_idx in tqdm(range(len(frame_names))):
        img_path = img_paths[frame_idx]
        # ret, frame = cap.read()
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if not ret:
        #     break

        width, height = frame.shape[:2][::-1]
        if not if_init:

            predictor.load_first_frame(frame)
            if_init = True

            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            # Let's add a positive click at (x, y) = (210, 350) to get started

            # for labels, `1` means positive click and `0` means negative click
            # points = np.array([[660, 267]], dtype=np.float32)
            # labels = np.array([1], dtype=np.int32)

            # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
            # )

            # add bbox
            bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
            )

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            # print(all_mask.shape)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255

                all_mask = cv2.bitwise_or(all_mask, out_mask)

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame)
        # cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # cap.release()
    gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)

if __name__ == '__main__':

    video_dir = "data/byd/20240903_out_choose"
    output_dir = "./outputs/byd/bydlight_0923_camerpredict"
    os.makedirs(output_dir, exist_ok=True)
    spe_name = output_dir.split("/")[-1]
    # logger 
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # logger_path = './log'
    # os.makedirs(logger_path, exist_ok=True)
    logger_name = f"{formatted_time}_{spe_name}.log"
    logger_file = os.path.join(output_dir, logger_name)
    logger.add(logger_file)
    step = 1
    main(video_dir, output_dir, step, logger)