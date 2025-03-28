import cv2

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # 记录letterbox操作的参数
    letterbox_params = {
        'original_shape': shape,
        'new_shape': new_shape,
        'pad': (dw, dh),
        'scale': r
    }

    return img, letterbox_params

def reverse_letterbox(results, letterbox_params):
    original_shape = letterbox_params['original_shape']
    new_shape = letterbox_params['new_shape']
    pad = letterbox_params['pad']
    scale = letterbox_params['scale']
    
    # 对于2D边界框
    if 'boxes' in results:
        boxes = results['boxes']
        # 移除填充
        boxes[:, [0, 2]] -= pad[0]  # x方向
        boxes[:, [1, 3]] -= pad[1]  # y方向
        # 反向缩放
        boxes /= scale
        
    # 对于mask
    if 'masks' in results:
        masks = results['masks']
        # 裁剪掉填充部分
        h, w = new_shape
        masks = masks[:, int(pad[1]):h-int(pad[1]), int(pad[0]):w-int(pad[0])]
        # 调整大小到原始尺寸
        masks = cv2.resize(masks.transpose(1, 2, 0), original_shape[::-1]).transpose(2, 0, 1)
    
    return {'boxes': boxes, 'masks': masks}

if __name__ == '__main__':

    img = cv2.imread('./data/byd/1725294989885850.jpeg')
    img_preprocessed = letterbox(img)
    cv2.imwrite('./data/byd/1725294989885850_letterbox.jpeg', img_preprocessed)