import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 


names = ['bicycle','motorcycle','car'  
            ,'person'
            ,'rider' 
            ,'truck'
            ,'bus'
]
small_object_names = ['bicycle','motorcycle','person','rider',
                        'Bicycle','Motorcycle'
                        ,'Person'
                        ,'Rider',
                        'Cyclist',
                        'Pedestrian',
                      ]

class MaskProcessor:
    def __init__(self):
        super().__init__()
        self.image = None
        self.masks = None
        self.boxes = None
        self.labels = None
        self.height = None
        self.width = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_image(self, image_path: str) -> None:
        """加载图像"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def init_grounding_dino_model(self, model_id="IDEA-Research/grounding-dino-tiny") -> None:
        """初始化DINO模型"""
        self.grounding_processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
    
    def grounding_dino_predict(self, image, text, threshold=0.97) -> None:
        """DINO预测"""
        inputs = self.grounding_processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        return results
    
    def init_yolo_model(self) -> None:
        """初始化YOLO模型"""
        from ultralytics import YOLO
        self.yolo_model = YOLO("yolov8x-seg.pt")
        
    def yolo_predict(self, colorImage, retina_masks=True) -> None:
        """YOLO预测"""
        results = self.yolo_model.predict(colorImage, retina_masks=retina_masks)
        return results

    def yolo_preprocess(self, image, input_points,input_labels, input_boxes, input_bboxes,
                    logger, yolo_orimask_path,frame,height,width,the=0.5):
        results = self.yolo_predict(image,retina_masks=True)
        classnames_yolo = []
        results_output = {}
        masks_yolo_filter_draw = np.zeros((height,width))
        masks_yolo_filter_single = []
        if results[0].masks is not None:
            # results[0].save(filename=masks_yolo_save_path)
            masks = results[0].masks.data
            clsses = results[0].boxes.cls.cpu().tolist()
            names_yolo = results[0].names
            boxes = results[0].boxes
            xyxyn = boxes.xyxyn
            conf = boxes.conf # 置信度
            #获取 Mask
            # masks_xy = results[0].masks.xy
            for id, mask in enumerate(masks):
                x0, y0, x1, y1 = xyxyn[id].cpu().numpy()*[width,height,width,height]
                input_box = np.array([x0, y0, x1, y1]).astype(int) # 整数
                # mask_xy = masks_xy[id]
                masks_yolo_filter_draw_single = np.zeros((height,width))
                clss = clsses[id]
                if names_yolo[clss] in small_object_names or names_yolo[clss] in names:
                    if conf[id] < the:
                        continue
                    mask=mask.to(torch.float).cpu().numpy()
                    masks_yolo_filter_draw[mask==1] = id + 1
                    masks_yolo_filter_draw_single[mask==1] = id + 1
                    center = np.array([int((x0 + x1) / 2), int((y0 + y1) / 2)])
                    input_point = center.reshape(1,2)
                    # input_boxes.append(input_box.reshape(1,4))
                    masks_yolo_filter_single.append(masks_yolo_filter_draw_single)
                    input_boxes.append(input_box)
                    input_points.append(input_point)
                    input_label = np.ones([input_point.shape[0]])
                    input_labels.append(input_label)
                    input_bboxes.append([])
                    classnames_yolo.append(names_yolo[clss])
            input_boxes = np.array(input_boxes)
            classnames_yolo = np.array(classnames_yolo)
            masks_yolo_filter_single = np.array(masks_yolo_filter_single)
            results_output["labels"] = classnames_yolo
            results_output["boxes"] = input_boxes
            results_output["masks"] = masks_yolo_filter_draw
            results_output["masks_single"] = masks_yolo_filter_single
        return results_output, results, classnames_yolo
    
    def init_sam2_img_model(self, sam2_checkpoint="./checkpoints/sam2_hiera_large.pt", model_cfg="sam2_hiera_l.yaml") -> None:
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.sam2_image_predictor = SAM2ImagePredictor(sam2_image_model)
        
    def sam2_img_predict(self, image, input_boxes,point_coords=None,point_labels=None):
        """SAM2图片预测方法"""
        self.sam2_image_predictor.set_image(image)
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.sam2_image_predictor.predict(
            point_coords,
            point_labels,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks, scores, logits
    
    def init_sam2_video_model(self, sam2_checkpoint="./checkpoints/sam2_hiera_large.pt", model_cfg="sam2_hiera_l.yaml") -> None:
        self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    def init_sam2_video_state(self, img_path,video_path,offload_video_to_cpu=True, async_loading_frames=True):
        """初始化SAM2视频状态"""
        inference_state = self.sam2_video_predictor.init_state(video_path,img_path,
                                                               offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames)
        return inference_state
    
    def sam2_video_predict(self) -> None:
        """SAM2视频预测方法"""
        pass

    def process_masks(self, masks: np.ndarray, boxes: List[List[int]], labels: List[str]) -> None:
        """处理mask"""
        self.masks = masks
        self.boxes = boxes
        self.labels = labels

    def combine_masks(self) -> np.ndarray:
        """合并多个mask"""
        if self.masks is None:
            raise ValueError("没有可用的masks")
        combined_mask = np.zeros(self.masks[0].shape, dtype=np.uint8)
        for i, mask in enumerate(self.masks):
            combined_mask[mask] = i + 1
        return combined_mask

    def draw_masks(self, alpha: float = 0.5) -> np.ndarray:
        """在图像上绘制mask"""
        if self.image is None or self.masks is None:
            raise ValueError("图像或masks不可用")
        
        overlay = self.image.copy()
        for mask in self.masks:
            color = np.random.randint(0, 255, 3)
            overlay[mask] = color
        
        output = cv2.addWeighted(self.image, 1 - alpha, overlay, alpha, 0)
        return output

    def draw_boxes(self, thickness: int = 2) -> np.ndarray:
        """在图像上绘制边界框"""
        if self.image is None or self.boxes is None:
            raise ValueError("图像或边界框不可用")
        
        output = self.image.copy()
        for box, label in zip(self.boxes, self.labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return output

    def save_result(self, output_path: str) -> None:
        """保存处理结果"""
        if self.image is None:
            raise ValueError("没有可用的图像")
        
        result = self.draw_masks()
        result = self.draw_boxes(result)
        
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
