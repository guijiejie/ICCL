import cv2
import numpy as np

def __vertical_halfconv(inputs, kernel_size, bot=False):
    outputs = np.zeros(inputs.shape)
    tmp = np.pad(inputs, ((0 if bot else kernel_size, kernel_size if bot else 0), (0, 0)))
    height = inputs.shape[0]
    for i in range(kernel_size):
        outputs += tmp[i:i+height, :]
    
    return (outputs >= 1).astype(np.uint8)

def horizontal_lines_detector_np(img, blur=(5, 1), canny=(40, 50), threshold=0.6, global_threshold=0.5, border_threshold=(0.6, 0.5), min_height_scale=0.3):
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)

    if len(img.shape) > 2:
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayimg = img
    grayimg = cv2.GaussianBlur(grayimg, blur, 0)

    height, width = img.shape[0:2]

    edge = cv2.Canny(grayimg, canny[0], canny[1])
    # edge = __vertical_halfconv(edge, 3, bot=False)

    accumulator = (edge > 0).sum(axis=1)[np.newaxis,:]
    

    M = accumulator.max()

    threshold = int(M*0.45) if threshold is None else int(M*threshold)
    threshold = max(threshold, width * global_threshold)

    result = np.array(np.where(accumulator > threshold))[1]
    ans = np.zeros(2)
    ans[1] = height
    for yaxis in result:
        if yaxis / height < border_threshold[0]:
            ans[0] = yaxis
            break

    for yaxis in result[::-1]:
        if (height - yaxis) / height < border_threshold[1]:
            ans[1] = yaxis
            break

    if (ans[1] - ans[0]) / height < min_height_scale:
        if width < height:
            ans[0] = (height - width) / 2.0
            ans[1] = width + (height - width) / 2.0
        else:
            ans[0] = 0.0
            ans[1] = height
    
    return ans, grayimg, edge

def vertical_lines_detector_np(img, blur=(1, 5), canny=(40, 50), threshold=0.7, global_threshold=0.7, border_threshold=(0.4, 0.4), min_width_scale=0.4):
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)
    if len(img.shape) > 2:
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayimg = img

    grayimg = cv2.GaussianBlur(grayimg, blur, 0)

    height, width = img.shape[0:2]

    edge = cv2.Canny(grayimg, canny[0], canny[1])
    accumulator = (edge > 0).sum(axis=0)[np.newaxis,:]

    M = accumulator.max()

    threshold = int(M*0.45) if threshold is None else int(M*threshold)
    threshold = max(threshold, height * global_threshold)

    result = np.array(np.where(accumulator > threshold))[1]
    
    ans = np.zeros(2)
    ans[1] = width
    for xaxis in result:
        if xaxis / width < border_threshold[0]:
            ans[0] = xaxis
            break

    for xaxis in result[::-1]:
        if (width - xaxis) / width < border_threshold[1]:
            ans[1] = xaxis
            break

    if (ans[1] - ans[0]) / width < min_width_scale:
        if height < width:
            ans[0] = (width - height) / 2.0
            ans[1] = height + (width - height) / 2.0
        else:
            ans[0] = 0.0
            ans[1] = width
    
    return ans, grayimg, edge

def lines_detector_np(img, intermediate_result=False):
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yborder, grayimg, edge = horizontal_lines_detector_np(grayimg, blur=(5, 1), canny=(40, 50), threshold=0.6, global_threshold=0.5, border_threshold=(0.6, 0.5), min_height_scale=0.3)
    xborder, grayimg, edge = vertical_lines_detector_np(grayimg, blur=(1, 5), canny=(40, 50), threshold=0.7, global_threshold=0.7, border_threshold=(0.4, 0.4), min_width_scale=0.4)

    if intermediate_result:
        return (xborder[0],yborder[0], xborder[1], yborder[1]), grayimg, edge
    else:
        return (xborder[0],yborder[0], xborder[1], yborder[1])