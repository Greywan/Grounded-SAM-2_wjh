

import numpy as np
import os

# yolo
from ultralytics import YOLO
model_yolo = YOLO("yolov8x-seg.pt") # 加载 YOLOv8n Segmentation 模型

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

def yolo_preprocess(colorImage, input_points,input_labels, input_boxes, input_bboxes,
                    logger, yolo_orimask_path,frame, height,width, the=0.5):
                    
    results = model_yolo.predict(colorImage)
    # masks_yolo = []
    classnames_yolo = []
    results_output = {}
    # masks_yolo_filter_draw = np.zeros((height,width))
    # masks_yolo_save_path = os.path.join(yolo_orimask_path,  f"{frame}" + '.jpg')
    if results[0].masks is not None:
        # results[0].save(filename=masks_yolo_save_path)
        masks = results[0].masks.data
        clsses = results[0].boxes.cls.cpu().tolist()
        names_yolo = results[0].names
        boxes = results[0].boxes
        xyxyn = boxes.xyxyn
        conf = boxes.conf # 置信度
        #获取 Mask
        masks_xy = results[0].masks.xy
        for id in range(0,len(masks)):
            x0, y0, x1, y1 = xyxyn[id].cpu().numpy()*[width,height,width,height]
            input_box = np.array([x0, y0, x1, y1]).astype(int) # 整数
            mask_xy = masks_xy[id]
            clss = clsses[id]
            if names_yolo[clss] in small_object_names or names_yolo[clss] in names:
                if conf[id] < the:
                    continue
                mask=np.zeros((height,width))
                if len(mask_xy)==0:
                    logger.info("mask_xy is none!!!\n")
                    continue
                # masks_yolo_filter_draw[mask==1] = id + 1
                center = np.array([int((x0 + x1) / 2), int((y0 + y1) / 2)])
                input_point = center.reshape(1,2)
                # input_boxes.append(input_box.reshape(1,4))
                input_boxes.append(input_box)
                input_points.append(input_point)
                input_label = np.ones([input_point.shape[0]])
                input_labels.append(input_label)
                input_bboxes.append([])
                classnames_yolo.append(names_yolo[clss])
        input_boxes = np.array(input_boxes)
        results_output["labels"] = classnames_yolo
        results_output["boxes"] = input_boxes
    return results_output, results, classnames_yolo