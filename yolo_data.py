import os
import cv2
import torch
import numpy as np
# import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import pdb
import time
from tqdm import tqdm
# import sys
# sys.path.append('./wjh')
from yolo.yolo_function_single import yolo_preprocess
from utils.calculate_time import splitting_time_eachnum,TimeCalculator
from loguru import logger
from datetime import datetime
from utils.vis import draw_box_specifycolor, show_sam2_masks, combine_masks,draw_sam2_masksboxes,draw_mask_specifycolor

def main(video_dir, output_dir, step):
    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = "car,person."

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
    # 'output_dir' is the directory to save the annotated frames
    # 'output_video_path' is the path to save the final video
    output_video_path = os.path.join(output_dir, "output.mp4")
    # create the output directory
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    yolo_orimask_path = os.path.join(output_dir, 'mask_yolo_ori')
    os.makedirs(yolo_orimask_path, exist_ok=True)
    yolo_filtermask_path = os.path.join(output_dir, 'mask_yolo_filter')
    CommonUtils.creat_dirs(yolo_filtermask_path)
    mask_result_dir = os.path.join(output_dir, "mask_result")
    CommonUtils.creat_dirs(mask_result_dir)
    cropped_img_dir = os.path.join(output_dir, "cropped_img")
    CommonUtils.creat_dirs(cropped_img_dir)
    cropped_mask_dir = os.path.join(output_dir, "cropped_mask")
    CommonUtils.creat_dirs(cropped_mask_dir)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    img_paths = [os.path.join(video_dir, frame_name) for frame_name in frame_names]

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    # height = inference_state["video_height"]
    # width = inference_state["video_width"]

    height = 576
    width = 1024

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    # for start_frame_idx in range(0, len(frame_names), step):
    # times = {}
    # times['time_frame_all'] = 0
    timer = TimeCalculator()

    time_start = time.time()
    for frame_idx in tqdm(range(len(frame_names))):
        time_single_start = time.time()

        start_frame_idx = frame_idx
        # print("start_frame_idx", start_frame_idx)
    
    # prompt grounding dino to get the box coordinates on specific frame
        print("start_frame_idx", start_frame_idx)
        # continue
        img_path = img_paths[start_frame_idx]
        image = Image.open(img_path)

        image = cv2.resize(np.array(image), (1024, 576))
        image = Image.fromarray(image)

        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_path = os.path.join(mask_result_dir, f"{frame_idx}_{image_base_name}" + '.jpg')
    
        # run YOLOv8-seg on the image
        input_points = []
        input_labels = []
        input_boxes = []
        input_bboxes = []
        
        colorImage = np.array(image.convert("RGB"))
        timer.start('yolo_preprocess_all')
        results_filteroutput, results_yolo, classnames_yolo = yolo_preprocess(colorImage[:, :, ::-1], input_points,input_labels, input_boxes, input_bboxes,
                    logger, yolo_orimask_path, image_base_name, height, width, the=0.5)
        masks_yolo_save_path = os.path.join(yolo_orimask_path,  f"{frame_idx}_{image_base_name}" + '.jpg')
        results_yolo[0].save(filename=masks_yolo_save_path)

        yolo_masks_cropped_single = results_filteroutput["cropped_masks"]
        for i, yolo_mask_single in enumerate(yolo_masks_cropped_single):
            cropped_img = results_filteroutput["cropped_imgs"][i]
            cropped_img_path = os.path.join(cropped_img_dir, f"{frame_idx}_{image_base_name}_{i}" + '.png')
            cv2.imwrite(cropped_img_path, cropped_img)
            cropped_mask = yolo_mask_single
            yolo_filtermask_single_path = os.path.join(cropped_mask_dir, f"{frame_idx}_{image_base_name}_{i}" + '.png')
            cv2.imwrite(yolo_filtermask_single_path, cropped_mask)

        #     yolo_mask_single_draw = np.zeros((576, 1024, 3), dtype=np.uint8)
        #     
        #     yolo_mask_single_draw = draw_mask_specifycolor(yolo_mask_single_draw, yolo_mask_single, font_scale=0.5)
        #     yolo_filtermask_single_path = os.path.join(mask_yolo_single_dir, f"{frame_idx:04d}_{image_base_name}_{obj_id}_{OBJECTS[i]}" + '.jpg')
        #     cv2.imwrite(yolo_filtermask_single_path, yolo_mask_single_draw)

        if len(classnames_yolo) == 0:
            logger.info(f"{image_base_name} : There is no object detected(yolo filtered)")
            # cv2.imwrite(mask_path, colorImage[:, :, ::-1])
            cv2.imwrite(mask_path, colorImage)
            continue
        else:
        # yolo_predict = True
            input_boxes = results_filteroutput["boxes"] # .cpu().numpy()
            yolo_filtermask_single_path = os.path.join(yolo_filtermask_path, f"{frame_idx}_{image_base_name}" + '.jpg')
            yolo_filtermask_draw = draw_box_specifycolor(colorImage[:, :, ::-1], input_boxes,font_scale=0.4)
            cv2.imwrite(yolo_filtermask_single_path, yolo_filtermask_draw)
        timer.end('yolo_preprocess_all')

        # process the detection results
        
        # # print("results[0]",results[0])
        # OBJECTS = results_filteroutput["labels"]
        # timer.start('sam2_img_predict_all')
        # # prompt SAM image predictor to get the mask for the object
        # image_predictor.set_image(colorImage)
        # # prompt SAM 2 image predictor to get the mask for the object
        # masks, scores, logits = image_predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_boxes,
        #     multimask_output=False,
        # )
        # # convert the mask shape to (n, H, W)
        # if masks.ndim == 2:
        #     masks = masks[None]
        #     scores = scores[None]
        #     logits = logits[None]
        # elif masks.ndim == 4:
        #     masks = masks.squeeze(1)
        # timer.end('sam2_img_predict_all')

        # # time_single_cost = time.time() - time_single_start
        # # times['time_frame_all'] += time_single_cost
        # # colorImage_Image = Image.fromarray(colorImage)
        # mask_img_combine = combine_masks(masks)
        # # mask_img_result = draw_sam2_masksboxes(colorImage[:, :, ::-1], mask_img_combine, input_boxes,font_scale=0.6)
        # mask_img_result = draw_mask_specifycolor(colorImage[:, :, ::-1], mask_img_combine, font_scale=0.6)
        # cv2.imwrite(mask_path, mask_img_result)
        # plt_mask = show_sam2_masks(colorImage[:, :, ::-1], masks, input_boxes)
        
        # plt_mask.savefig(mask_path)

    # time_cost = time.time() - time_start


    times = timer.splitting_time_eachnum(len(frame_names), "all", "every_frame")
    for key, value in times.items():
        logger.info(f"{key}: {value:.4f} second")
if __name__ == '__main__':

    video_dir = "data/byd/20240903_out_choose"
    output_dir = "./outputs/yolo_data"
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
    main(video_dir, output_dir, step)