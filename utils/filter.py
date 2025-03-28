

import numpy as np
import os
import cv2
from PIL import Image
import time
def filter_overlapping_masks(mask_lists, overlap_threshold=0.5, containment_threshold=0.8):
    """
    过滤重叠的mask，保留较大的mask。考虑包含关系的情况。
    
    参数:
    mask_lists: numpy数组，形状为(n, H, W)
    overlap_threshold: 重叠阈值，默认为0.5
    containment_threshold: 包含阈值，默认为0.8
    
    返回:
    filtered_masks: 过滤后的mask列表
    """
    n, h, w = mask_lists.shape
    filtered_masks_indices = []
    
    # 计算每个mask的面积
    areas = np.sum(mask_lists, axis=(1, 2))
    
    # 按面积从大到小排序
    sorted_indices = np.argsort(areas)[::-1]
    
    for i in sorted_indices:
        keep = True
        for j in filtered_masks_indices:
            intersection = np.logical_and(mask_lists[i], mask_lists[j])
            intersection_area = np.sum(intersection)
            
            # 计算IoU
            union = np.logical_or(mask_lists[i], mask_lists[j])
            iou = intersection_area / np.sum(union)
            
            # 计算包含率
            containment_ratio_i = intersection_area / areas[i]
            containment_ratio_j = intersection_area / areas[j]
            
            if iou > overlap_threshold or containment_ratio_i > containment_threshold or containment_ratio_j > containment_threshold:
                keep = False
                break
        
        if keep:
            filtered_masks_indices.append(i)
    
    return filtered_masks_indices

def filter_contained_masks(mask_lists, overlap_threshold=0.5, containment_threshold=0.8):
    """
    过滤包含关系的mask，保留较大的mask。
    
    参数:
    mask_lists: numpy数组，形状为(n, H, W)
    containment_threshold: 包含阈值，默认为0.8
    
    返回:
    filtered_masks: 过滤后的mask列表
    """
    # n, h, w = mask_lists.shape
    # filtered_masks_indices = []
    
    # 计算每个mask的面积
    areas = np.sum(mask_lists, axis=(1, 2))
    
    # 按面积从大到小排序
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算当前框与其他框的交集
        intersections = np.logical_and(mask_lists[i], mask_lists[order[1:]])
        intersections_area = np.sum(intersections, axis=(1, 2))

        unions = np.logical_or(mask_lists[i], mask_lists[order[1:]])
        ious = intersections_area / np.sum(unions)
    
        # 根据包含率和 iou 过滤
        containment_ratio = intersections_area / areas[order[1:]]
        inds = np.where((containment_ratio <= containment_threshold) & (ious <= overlap_threshold))[0]
        order = order[inds + 1]
    
    return keep
    


def filter_overlapping_boxes(boxes, containment_threshold=0.78):
    """
    过滤重叠的2D边界框，只滤除有包含关系的小框。
    
    参数:
    boxes: numpy数组，形状为(n, 4)，每行表示一个边界框 [x1, y1, x2, y2]
    containment_threshold: 包含阈值，默认为0.8
    
    返回:
    filtered_boxes: 过滤后的边界框列表
    """
    if len(boxes) == 0:
        return np.array([])
    
    # 计算每个框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按面积从大到小排序
    order = areas.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算当前框与其他框的交集
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # 计算包含率
        containment_ratio = inter / areas[order[1:]]
        
        # 只根据包含率过滤
        inds = np.where(containment_ratio <= containment_threshold)[0]
        order = order[inds + 1]
    
    return keep

# 使用示例
# boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]])
# filtered_boxes = filter_overlapping_boxes(boxes)
# print(f"原始边界框数量: {len(boxes)}")
# print(f"过滤后边界框数量: {len(filtered_boxes)}")


# 使用示例
# mask_lists = np.random.randint(0, 2, (10, 100, 100))
# filtered_masks = filter_overlapping_masks(mask_lists)
# print(f"原始mask数量: {mask_lists.shape[0]}")
# print(f"过滤后mask数量: {filtered_masks.shape[0]}")

def read_masks(img_dir):
    masks = []
    img_paths = os.listdir(img_dir)
    for path in img_paths:
         
        img_path = os.path.join(img_dir, path)
        # img = Image.open(img_path)
        mask = cv2.imread(img_path)
        
        mask_prepare = np.zeros(mask.shape[:2])
        
        mask_prepare[mask[:,:,0]==255] = 1
        masks.append(mask_prepare)
    masks = np.array(masks)
    return masks

if __name__ == "__main__":
    img_paths = './data/debug'
    masks = read_masks(img_paths)
    masks_path = './data/0007_1725294991485849.npy'
    masks = np.load(masks_path)
    time_start = time.time()
    filter_indexs = filter_contained_masks(masks)
    time_end = time.time()
    print(time_end - time_start)
    print(filter_indexs)

