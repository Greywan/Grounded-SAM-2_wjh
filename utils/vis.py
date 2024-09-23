import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

VIS_COLOR_MAP = {
    -1: [109, 49, 50],  # [0,0,0], 
    0: [109, 49, 50],  #[123, 79, 152],
    1: [250, 70, 70],
    2: [250, 70, 147],
    3: [6, 230, 230],
    4: [78, 125, 98],
    5: [128, 64, 128],
    6: [70, 200, 200],
    7: [93, 71, 139],
    8: [148, 0, 211],
    9: [139, 115, 85],
    10: [119, 11, 32],
    11: [180, 50, 30],
    12: [250, 218, 141],
    13: [25, 25, 112],
    14: [250, 218, 141],
    15: [240, 255, 240],
    16: [119, 11, 32],
    17: [93, 71, 139],
    18: [107, 142, 35],
    19: [255, 114, 86],
    20: [148, 0, 211],
    21: [20, 60, 255],
    22: [65, 105, 225],
    23: [25, 25, 112],
    24: [240, 255, 240],
    25: [0, 191, 255],
    26: [162, 205, 90],
    27: [25, 25, 112],
    28: [250, 128, 10],
    29: [12, 25, 136],
    30: [78, 125, 98],
    31: [90, 58, 200],
    32: [68, 71, 21],
    33: [162, 205, 90],
    34: [25, 25, 112],
    35: [250, 128, 10],
    36: [12, 25, 136],
    37: [78, 125, 98],
    38: [90, 58, 200],
    39: [78, 125, 98],
    40: [90, 58, 200],
    255: [255, 255, 255]
}

def draw_box_specifycolor(img,boxes,box_scale=2,radius=3,font_scale=1.2,thickness=2):
    img_draw = img.copy()
    color_seed = (255, 255, 0) #黄色
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        center_x = int((x0 + x1) / 2)
        center_y = int((y0 + y1) / 2)
        if idx < 40:
            draw_color = np.array(VIS_COLOR_MAP[idx]).astype(np.uint8).tolist()
        else:
            draw_color = np.array(VIS_COLOR_MAP[idx%40]).astype(np.uint8).tolist()
        # draw_color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        # draw_color = np.array(VIS_COLOR_MAP[idx]).astype(np.uint8).tolist()
        cv2.rectangle(img_draw, (x0, y0), (x1, y1), draw_color, box_scale)
        cv2.circle(img_draw, (center_x, center_y), radius, color_seed, -1)
        cv2.putText(img_draw, str(idx), (center_x, center_y), font, font_scale, draw_color, thickness)
    return img_draw

def show_box(box, ax, image_width, image_height):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x0 + w, image_width)
    y1 = min(y0 + h, image_height)
    # 如果调整后宽度或高度为0，则不绘制矩形
    if x1 <= x0 or y1 <= y0:
        return
    # 更新边界框尺寸
    w = x1 - x0
    h = y1 - y0
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='green',
                      facecolor=(0, 0, 0, 0),
                      lw=2))

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
            for contour in contours
        ]
        mask_image = cv2.drawContours(mask_image,
                                      contours,
                                      -1, (1, 1, 1, 0.5),
                                      thickness=2)
    ax.imshow(mask_image)

def draw_mask_specifycolor(img,masks,font_scale=1.2):
    img_draw = img.copy()
    labels = np.unique(masks)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 0) #黄色
    thickness = 2
    color_seed = (0, 255, 255)
    radius = 3

    for label in labels:
        if label == 0:
            continue
        else:
            if label < 40:
                draw_color = np.array(VIS_COLOR_MAP[label])
            else:
                draw_color = np.array(VIS_COLOR_MAP[label%40])
            # draw_color = (_COLORS[label] * 255).astype(np.uint8).tolist()
            # draw mask
            filter_mask=(masks==label)
            pos = np.where(masks == label)
            choose = len(pos[0]) // 2
            pos_x = pos[0][choose]
            pos_y = pos[1][choose]
            img_draw[filter_mask] = img[filter_mask] *0.2 + np.array(draw_color)*0.8
            cv2.putText(img_draw, str(int(label)), (pos_y, pos_x), font, font_scale, color, thickness)
            cv2.circle(img_draw, (pos_y, pos_x), radius, color_seed, -1)
    return img_draw

def draw_box_specifycolor(img,boxes,box_scale=2,radius=3,font_scale=1.2,thickness=2):
    img_draw = img.copy()
    color_seed = (255, 255, 0) #黄色
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        center_x = int((x0 + x1) / 2)
        center_y = int((y0 + y1) / 2)
        # draw_color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        draw_color = np.array(VIS_COLOR_MAP[idx]).astype(np.uint8).tolist()
        cv2.rectangle(img_draw, (x0, y0), (x1, y1), draw_color, box_scale)
        cv2.circle(img_draw, (center_x, center_y), radius, color_seed, -1)
        cv2.putText(img_draw, str(idx), (center_x, center_y), font, font_scale, draw_color, thickness)
    return img_draw

def show_sam2_masks(colorImage, masks, boxes):
    plt.clf()
    # 设置图形大小
    plt.figure(figsize=(colorImage.size[0]/100, colorImage.size[1]/100))
    # 显示原始图像
    plt.imshow(colorImage)
    for idx, mask in enumerate(masks):
        box = boxes[idx]
        show_box(box, plt.gca(), colorImage.size[0], colorImage.size[1])
        show_mask(mask, plt.gca(), random_color=False)
    plt.axis('off')
    # 移除图形周围的空白边距
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    return plt

def combine_masks(mask_list, background_value = 0):
    mask_img = np.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        final_index = background_value + idx + 1
        if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
            raise ValueError("The mask shape should be the same as the mask_img shape.")
        # mask = mask
        mask_img[mask == True] = final_index
    return mask_img

def draw_sam2_masksboxes(colorImage, masks, boxes,font_scale=1.2):
    img_draw = draw_mask_specifycolor(colorImage, masks, font_scale=font_scale)
    img_draw = draw_box_specifycolor(img_draw, boxes, font_scale=font_scale)
    return img_draw

def draw_mask_fromnpy(file, path, name, font_scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 0) #黄色
    thickness = 2

    masks = np.load(file)
    labels = np.unique(masks)
    for label in labels:
        if label == 0:
            continue
        else:
            if label < 40:
                draw_color = np.array(VIS_COLOR_MAP[label])
            else:
                draw_color = np.array(VIS_COLOR_MAP[label%40])
            # draw_color = (_COLORS[label] * 255).astype(np.uint8).tolist()
            # draw mask
            filter_mask=(masks==label)
            mask_draw = np.zeros((masks.shape[0], masks.shape[1], 3))
            pos = np.where(masks == label)
            choose = len(pos[0]) // 2
            pos_x = pos[0][choose]
            pos_y = pos[1][choose]
            mask_draw[filter_mask] = np.array(draw_color)*0.8
            cv2.putText(mask_draw, str(int(label)), (pos_y, pos_x), font, font_scale, color, thickness)
            img_path = os.path.join(path, name + f'_{label}.png')
            cv2.imwrite(img_path, mask_draw)
