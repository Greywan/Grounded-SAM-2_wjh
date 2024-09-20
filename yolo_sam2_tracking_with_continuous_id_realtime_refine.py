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
from yolo.yolo_function import yolo_preprocess
from utils.calculate_time import splitting_time_eachnum
from loguru import logger
from datetime import datetime
from utils.vis import draw_box_specifycolor,draw_mask_specifycolor
from mask_propocess import MaskProcessor
def main(video_dir, output_dir, step, logger):
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
    maskProcessor = MaskProcessor()
    maskProcessor.init_sam2_video_model(sam2_checkpoint, model_cfg)
    maskProcessor.init_sam2_img_model(sam2_checkpoint, model_cfg)
    sam2_video_predictor = maskProcessor.sam2_video_predictor

    maskProcessor.init_yolo_model()
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
    mask_result_dir = os.path.join(output_dir, "mask_result")
    CommonUtils.creat_dirs(mask_result_dir)
    mask_single_dir = os.path.join(output_dir, "mask_single")
    CommonUtils.creat_dirs(mask_single_dir)

    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    yolo_orimask_path = os.path.join(output_dir, 'mask_yolo_ori')
    os.makedirs(yolo_orimask_path, exist_ok=True)
    yolo_filtermask_path = os.path.join(output_dir, 'mask_yolo_filter')
    CommonUtils.creat_dirs(yolo_filtermask_path)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    img_paths = [os.path.join(video_dir, frame_name) for frame_name in frame_names]
    # init video predictor state
    inference_state = maskProcessor.init_sam2_video_state(img_path=img_paths[0], 
                                                          video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    inference_state["video_height"] = 576
    inference_state["video_width"] = 1024
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    # for start_frame_idx in range(0, len(frame_names), step):
    times = {}
    times['time_frame_all'] = 0

    time_start = time.time()
    for frame_idx in tqdm(range(len(frame_names))):
        time_single_start = time.time()

        if frame_idx % step != 0: # 非 step 的帧跳过
            continue
        else:
            start_frame_idx = frame_idx
            print("start_frame_idx", start_frame_idx)
    
    # prompt grounding dino to get the box coordinates on specific frame
        print("start_frame_idx", start_frame_idx)
        # continue
        img_path = img_paths[start_frame_idx]
        image = Image.open(img_path)

        image = cv2.resize(np.array(image), (1024, 576))
        image = Image.fromarray(image)

        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
        # mask_path = os.path.join(sam_mask_path, f"{image_base_name}" + '.jpg')
        mask_path = os.path.join(mask_result_dir, f"{frame_idx}_{image_base_name}" + '.jpg')
        # run Grounding DINO on the image
        # inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        # with torch.no_grad():
        #     outputs = grounding_model(**inputs)

        # results = processor.post_process_grounded_object_detection(
        #     outputs,
        #     inputs.input_ids,
        #     box_threshold=0.25,
        #     text_threshold=0.25,
        #     target_sizes=[image.size[::-1]]
        # )
        
        # run YOLOv8-seg on the image
        input_points = []
        input_labels = []
        input_boxes = []
        input_bboxes = []
        
        colorImage = np.array(image.convert("RGB"))
        
        yolo_outputs = maskProcessor.yolo_preprocess(colorImage[:, :, ::-1], input_points,input_labels,
                                                    input_boxes, input_bboxes,logger, yolo_orimask_path, 
                                                    frame_idx, height, width,the=0.5)
                                                    
        results_filteroutput, results_yolo, classnames_yolo = yolo_outputs

        masks_yolo_save_path = os.path.join(yolo_orimask_path,  f"{frame_idx}_{image_base_name}" + '.jpg')
        results_yolo[0].save(filename=masks_yolo_save_path)

        if len(classnames_yolo) == 0:
            logger.info(f"{image_base_name} : There is no object detected(yolo filtered)")
            cv2.imwrite(mask_path, colorImage)
            continue
        # else:
        yolo_predict = True
        input_boxes = results_filteroutput["boxes"] # .cpu().numpy()
        yolo_filtermask_single_path = os.path.join(yolo_filtermask_path, f"{frame_idx}_{image_base_name}" + '.jpg')
        yolo_filtermask_draw = draw_box_specifycolor(colorImage[:, :, ::-1], input_boxes,font_scale=0.4)
        cv2.imwrite(yolo_filtermask_single_path, yolo_filtermask_draw)
        

        # process the detection results
        
        # print("results[0]",results[0])
        OBJECTS = results_filteroutput["labels"]
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = maskProcessor.sam2_img_predict(image, input_boxes)
        """
        Step 3: Register each object's positive points to video predictor
        """
        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")

        #debug vis
        mask_2d_dict = copy.deepcopy(mask_dict)
        
        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        # if frame_idx == 43:
        #     import ipdb; ipdb.set_trace()
        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.3, objects_count=objects_count)
        print("objects_count", objects_count)
    
        # video_predictor.reset_state(inference_state)
        if len(mask_dict.labels) == 0:
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        
        
        sam2_video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = sam2_video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                    img_path
                )
        
        #debug vis
        # for object_id, object_info in mask_2d_dict.labels.items():
        #     obj_idx = object_id
        #     obj_mask = object_info.mask.cpu().numpy()
        #     # if obj_mask.sum() < 30:
        #     #     continue
        #     mask_vis = np.zeros((576, 1024), dtype=np.uint8)
        #     mask_vis[obj_mask == True] = 255
        #     mask_vis_save_path = os.path.join(mask_single_dir, f"{frame_idx}_{image_base_name}_{obj_idx}_2dmask" + '.jpg')
        #     cv2.imwrite(mask_vis_save_path, mask_vis)
        
        # for object_id, object_info in sam2_masks.labels.items():
        #     # if len(inference_state["obj_id_to_idx"]) >0:
        #     #     obj_idx = inference_state["obj_id_to_idx"].get(object_id, None)
        #     # else:
        #     obj_idx = object_id
        #     obj_mask = object_info.mask.cpu().numpy()
        #     mask_vis = np.zeros((576, 1024), dtype=np.uint8)
        #     mask_vis[obj_mask == True] = 255
            # mask_vis_save_path = os.path.join(mask_single_dir, f"{frame_idx}_{image_base_name}_{obj_idx}_videomask" + '.jpg')
            # cv2.imwrite(mask_vis_save_path, mask_vis)
        # debug vis

        video_segments = {}  # output the following {step} frames tracking masks
        (processing_order, clear_non_cond_mem, 
         batch_size) = sam2_video_predictor.propagate_in_video_prepare_wjh(inference_state, 
                                                                                         img_paths, max_frame_num_to_track=step, start_frame_idx=start_frame_idx)
        
        for out_frame_idx in tqdm(processing_order, desc="propagate in video"):
            img_path = img_paths[out_frame_idx]
            output = sam2_video_predictor.propagate_in_video_wjh(inference_state, 
                                                            out_frame_idx, img_path, clear_non_cond_mem,batch_size)
            out_obj_ids, out_mask_logits = output
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                if out_mask.sum() < 30:
                    continue
                object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        time_single_cost = time.time() - time_single_start
        logger.info(f"{frame_idx}_{image_base_name} time cost:{time_single_cost}.")

        times['time_frame_all'] += time_single_cost

        print("video_segments:", len(video_segments))
        """
        Step 5: save the tracking masks and json files
        """
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            frame_name = frame_names[frame_idx].split(".")[0]

            for obj_id, obj_info in mask.items():
                mask_where = obj_info.mask == True
                if mask_where.sum() < 30:
                    logger.info(f"{frame_name}_{obj_id} is too small, skip it.")
                    continue
                mask_img[mask_where] = obj_id
                instance_id = obj_info.get_id()
                # mask_draw = np.zeros((frame_masks_info.mask_height, frame_masks_info.mask_width), dtype=np.uint8)
                # mask_draw[obj_info.mask == True] = 255
                # mask_single_path = os.path.join(mask_data_dir, frame_names[frame_idx].split(".")[0] + f'{obj_id}.jpg')
                # cv2.imwrite(mask_single_path, mask_draw)

            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            json_data = frame_masks_info.to_dict()
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)
            #vis
            img_path = img_paths[frame_idx]
            img_draw = cv2.imread(img_path)
            img_draw = cv2.resize(img_draw, (1024, 576))
            img_draw = draw_mask_specifycolor(img_draw, mask_img,font_scale=0.4)
            cv2.imwrite(os.path.join(mask_result_dir, f"{frame_idx}_{frame_name}" + '.jpg'), img_draw)


    time_cost = time.time() - time_start
    times = splitting_time_eachnum(times, len(frame_names), "all", "every_frame")
    for key, value in times.items():
        logger.info(f"{key}: {value:.2f} second")
    """
    Step 6: Draw the results and save the video
    """
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

    create_video_from_images(result_dir, output_video_path, frame_rate=15)

    
if __name__ == '__main__':

    video_dir = "data/byd/20240903_out_choose"
    output_dir = "./outputs/byd/bydlight_refinetest"
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