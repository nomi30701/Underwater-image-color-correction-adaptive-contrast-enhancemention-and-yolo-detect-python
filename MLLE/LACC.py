import cv2
import numpy as np

def histogram_spread(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 1))
    return np.std(hist)

def LACC(input_img: np.ndarray, is_vid=False, is_run=False):
    """
    Locally Adaptive Color Correction.

    Parameters:
    - img (numpy.ndarray): a 3-channel color image (BGR). values in the range [0, 1]
    - is_vid (bool): check is it video
    - is_run (bool): check is it first run. (for video)

    Returns:
    - LACC_img (numpy.ndarray): color corrected img. values in the range [0, 1]
    - is_run
    """

    ## zip [(img_mean, img)], it (b, g, r)
    small, medium, large = sorted(list(zip(cv2.mean(input_img), cv2.split(input_img), ['b', 'g', 'r'])))
    ## sorted by mean (small to large)
    small, medium, large = list(small), list(medium), list(large)
    
    ## exchange wrong channel
    if is_vid and not is_run:
        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':
            large, medium = medium, large
            print('exchange!')
        is_run = True
        
    elif not is_vid:
        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':
            large, medium = medium, large

    ## Max attenuation
    max_attenuation = 1 - (small[1]**1.2)
    max_attenuation = np.expand_dims(max_attenuation, axis=2)

    ## Detail image
    blurred_image = cv2.GaussianBlur(input_img, (7, 7), 0)
    detail_image = input_img - blurred_image
    
    ## corrected large channel
    large[1] = (large[1] - cv2.minMaxLoc(large[1])[0]) * (1/(cv2.minMaxLoc(large[1])[1] - cv2.minMaxLoc(large[1])[0]))
    large[0] = cv2.mean(large[1])[0]
    
    ## Iter corrected 
    loss = float('inf')
    while loss > 1e-2:
        medium[1] = medium[1] + (large[0] - cv2.mean(medium[1])[0]) * large[1]
        small[1] = small[1] + (large[0] - cv2.mean(small[1])[0]) * large[1]
        loss = abs(large[0] - cv2.mean(medium[1])[0]) + abs(large[0] - cv2.mean(small[1])[0])

    ## b, g, r combine
    for _, ch, color in [large, medium, small]:
        if color == 'b':
            b_ch = ch
        elif color == 'g':
            g_ch = ch
        else:
            r_ch = ch
    img_corrected = cv2.merge([b_ch, g_ch, r_ch])
    
    ## LACC Result
    LACC_img = detail_image + (max_attenuation * img_corrected) + ((1 - max_attenuation) * input_img)
    LACC_img = np.clip(LACC_img, 0.0, 1.0) 

    return LACC_img, is_run