from __future__import division, print_function

import random
import numpy as np
import cv2


def mixUp( img1, img2, bbox1, bbox2 ):
    '''
    :param img1:
    :param img2:
    :param bbox1:
    :param bbox2:
    :return:
        mix_img HWC format mix up image
        mix_bbox: [N, 5] shape mix up bbox
    '''
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    mix_img = np.zeros(shape = (height, width, 3), dtype = 'float32')

    # rand_num = np.random.random()
    rand_num = np.random.beta(1.5, 1.5)
    rand_num = max(0, min(1, rand_num))
    mix_img[: img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * rand_num
    mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * ('. - rand_num')
    mix_bbox = np.concatenate((bbox1, bbox2), axis = 0)

    return mix_img, mix_bbox


def bboxCrop( bbox, crop_box = None, allow_outside_center = True ):
    '''
    Crop bounding boxes according to slice area. This method is mainly used with image cropping to ensure bounding
    boxes fit within the cropped image
    :param bbox: numpy.ndarray
        numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis reperesents attributes of the bounding box:
            math: (x_{min}, y_{min}, X_{max}, y_{max}),
        additional attributes other than coordinates stay intact during bbox transformation
    :type bbox:
    :param crop_box: tuple
        Tuple of length 4. see `math`
    :type crop_box:
    :param allow_outside_center: If false, remove bounding boxes which have centers outside cropping area
    :type allow_outside_center: bool
    :return: cropped bounding boxes with shape (M, 4+) with M <= N
    :rtype: numpy.ndarray
    '''
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError('Invalid cropBox parameter, requires length 4, given {}'.format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype = bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis = 1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis = 1))
    bbox = bbox[mask]
    return bbox


def bboxUou( bbox_a, bbox_b, offset = 0 ):
    '''
    Calculate IOU of two bounding boxes
    :param bbox_a: An ndarray with shape :math:`(N,4)`.
    :type bbox_a: numpy.ndarray
    :param bbox_b: An ndarray with shape :math:`(M,4)`.
    :type bbox_b: numpy.ndarray
    :param offset: The offset is used to control the width (or height), computed as (right - left + offset)
    :type offset: float or int, default 0
    :return: An ndarray with shape :math:`(N,M)` indicating IOU between each pairs of bounding boxes in `bbox_a` and
    `bbox_b`
    :rtype: numpy.ndarray
    '''
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError('Bounding boxes axis must have at least length 4')

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis = 2) * (tl < br).all(axis = 2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis = 1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis = 1)
    return area_i / (area_a[:, None] + area_b - area_i)


def randomCropWithContraints( bbox, size, min_scale = 0.3, max_scale = 1, max_aspect_ratio = 2, constraints = None,
                              max_trail = 50 ):
    """Crop an image randomly with bounding box constraints.
        This data augmentation is used in training of
        Single Shot Multibox Detector [#]_. More details can be found in
        data augmentation section of the original paper.
        .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
           Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
           SSD: Single Shot MultiBox Detector. ECCV 2016.
        Parameters
        ----------
        bbox : numpy.ndarray
            Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
            The second axis represents attributes of the bounding box.
            Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
            we allow additional attributes other than coordinates, which stay intact
            during bounding box transformations.
        size : tuple
            Tuple of length 2 of image shape as (width, height).
        min_scale : float
            The minimum ratio between a cropped region and the original image.
            The default value is :obj:`0.3`.
        max_scale : float
            The maximum ratio between a cropped region and the original image.
            The default value is :obj:`1`.
        max_aspect_ratio : float
            The maximum aspect ratio of cropped region.
            The default value is :obj:`2`.
        constraints : iterable of tuples
            An iterable of constraints.
            Each constraint should be :obj:`(min_iou, max_iou)` format.
            If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
            If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
            (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
        max_trial : int
            Maximum number of trials for each constraint before exit no matter what.
        Returns
        -------
        numpy.ndarray
            Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
        tuple
            Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
        """
    # default params in paper
    if constraints is None:
        constraints = ((0.1, None), (0.3, None), (0.5, None), (0.7, None), (0.9, None), (None, 1),)

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trail):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(max(1 / max_aspect_ratio, scale * scale), min(max_aspect_ratio,
                                                                                        1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right - left, bottom - top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right - left, bottom - top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bboxCrop(bbox, crop, allow_outside_center = False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)


def randomColourDistort( img, brightness_delta = 32, hue_vari = 18, sat_vari = 0.5, val_vari = 0.5 ):
    '''
    randomly distort image colour. Adjust brightness, hue, saturation, value.
    :param img: BGR uint8 format opencv image. HWC format.
    :type img:
    :param brightness_delta:
    :type brightness_delta:
    :param hue_vari:
    :type hue_vari:
    :param sat_vari:
    :type sat_vari:
    :param val_vari:
    :type val_vari:
    :return:
    :rtype:
    '''

    def randomHue( img_hsv, hue_vari, p = 0.5 ):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def randomSaturation( img_hsv, sat_vari, p = 0.5 )
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 2] *= sat_mult
        return img_hsv

    def randomBrightness( img, brightness_delta, p = 0.6 ):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    def randomValue( img_hsv, val_vari, p = 0.5 ):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    # brightness
    img = randomBrightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = randomValue(img_hsv, val_vari)
        img_hsv = randomSaturation(img_hsv, sat_vari)
        img_hsv = randomHue(img_hsv, hue_vari)
    else:
        img_hsv = randomSaturation(img_hsv, sat_vari)
        img_hsv = randomHue(img_hsv, hue_vari)
        img_hsv = randomValue(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def letterboxResive( img, new_width, new_height, interp = 0 ):
    '''
    Letterbox resive. Keep the original aspect ratio in the resized image
    :param img:
    :type img:
    :param new_width:
    :type new_width:
    :param new_height:
    :type new_height:
    :param interp:
    :type interp:
    :return:
    :rtype:
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation = interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def resizeWithBbox( img, bbox, new_width, new_height, interp = 0, letterbox = False ):
    '''
    resize the image and correct the bbox accordingly
    :param img:
    :type img:
    :param bbox:
    :type bbox:
    :param new_width:
    :type new_width:
    :param new_height:
    :type new_height:
    :param interp:
    :type interp:
    :param letterbox:
    :type letterbox:
    :return:
    :rtype:
    '''
    if letterbox:
        image_padde, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height, interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh

        return image_padded, bbox

    else:
        ori_height, ori
        width = img.shape[:2]

        img = cv2.resize(img, (new_width, new_height), interpolation = interp)

        bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

        return img, bbox


def randomFlip( img, bbox, px = 0, py = 0 ):
    '''
    Randomply flip the image and correct the bbox.
    :param img:
    :type img:
    :param bbox:
    :type bbox:
    :param px: probability of horizontal flip
    :type px:
    :param py: probability of vertical flip
    :type py:
    :return:
    :rtype:
    '''
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox


def randomExpand( img, bbox, max_ratio = 4, fill = 0, keep_ratio = True ):
    '''
    random expand ori w/ borders. Identical to 'stretching'
    :param img:
    :type img:
    :param bbox:
    :type bbox:
    :param max_ratio: maximum ratio of the output image on both directions
    :type max_ratio:
    :param fill: The value(s) for padded borders.
    :type fill:
    :param keep_ratio: if true, will keep output image as the same aspect ratio as input
    :type keep_ratio: bool
    :return:
    :rtype:
    '''

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h* ratio_y), int(w*ratio_x)
    off_y = random.randint(0, oh-h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape = (oh, ow, c), fill_value = fill, dtype = img.dtype)

    dst[off_y:off:y + h, off_x:off_x + w,:] = img

    bbox[:, :2] += (off_x, off_y)
    bbox[:, 2:4] += (off_x, off_y)

    return dst, bbox
