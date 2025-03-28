import cv2
import numpy as np
import torch

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

    def init_sam2_video_state(self, img,video_path,offload_video_to_cpu=True, async_loading_frames=True):
        """初始化SAM2视频状态"""
        inference_state = self.sam2_video_predictor.init_state(video_path,img,
                                                               offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames)
        return inference_state
    
    def sam2_video_predict(self) -> None:
        """SAM2视频预测方法"""
        pass

    # def process_masks(self, masks: np.ndarray, boxes: List[List[int]], labels: List[str]) -> None:
    #     """处理mask"""
    #     self.masks = masks
    #     self.boxes = boxes
    #     self.labels = labels

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
