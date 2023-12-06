import cv2
import numpy as np

def histogram_spread(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 1))
    return np.std(hist)

def process_block(block, lc_variance, block_mean, block_variance, beta):
    
    alpha = ((lc_variance) / block_variance)
    if alpha < beta:
        block = block_mean + (alpha * (block - block_mean))
    else:
        block = block_mean + (beta * (block - block_mean))

    return block

def LACC(input_img: np.ndarray, is_vid=False, is_run=False) -> np.float64:
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

def LACE(input_img: np.ndarray, beta: float) -> np.uint8:
    """
    Locally Adaptive Contrast Enhancement.

    Parameters:
    - img (numpy.ndarray): a 3-channel color image (BGR). values in the range [0, 255]

    Returns:
    - LACE_img (numpy.ndarray): contrast enhancement img. values in the range [0, 255]
    """
    ## Process input image
    input_img = input_img.astype(np.uint8) 

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(input_img)

    ## Set parament
    block_size = 25
    beta = beta # Enhancement value
    radius = 10
    eps = 0.01
    
    ## Assuming l_channel
    lc_variance = np.var(l_channel)
    integral_sum, integral_sqsum = cv2.integral2(l_channel)
    height, width = l_channel.shape

    l_channel_processed = np.zeros_like(l_channel, dtype=np.float64)
    weight_sum = np.zeros_like(l_channel, dtype=np.float64)


    ## Process each block
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            ## Define block boundaries
            start_i = i
            end_i = min(i + block_size, height)
            start_j = j
            end_j = min(j + block_size, width)
            
            ## Extract block
            block = l_channel[start_i:end_i, start_j:end_j]

            ## Cal block var, mean
            block_sum = integral_sum[end_i, end_j] - integral_sum[start_i, end_j] - integral_sum[end_i, start_j] + integral_sum[start_i, start_j]
            block_mean = block_sum / ((end_i - start_i) * (end_j - start_j))
            block_sum_sq = integral_sqsum[end_i, end_j] - integral_sqsum[start_i, end_j] - integral_sqsum[end_i, start_j] + integral_sqsum[start_i, start_j]
            block_variance = block_sum_sq / ((end_i - start_i) * (end_j - start_j)) - np.square(block_mean)

            ## Process block
            block_processed = process_block(block, lc_variance, block_mean, block_variance, beta)
            
            ## Put block back into image
            l_channel_processed[start_i:end_i, start_j:end_j] += block_processed
            weight_sum[start_i:end_i, start_j:end_j] += 1.0

    l_channel_processed /= weight_sum
    l_channel_processed = np.clip(l_channel_processed, 0, 255).astype('uint8')
    
    ## guided (need install opencv-contrib-python)
    l_channel_processed = cv2.ximgproc.guidedFilter(l_channel, l_channel_processed, radius, eps)

    ## ab channel balance
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)
    if a_mean > b_mean:
        b_channel = (b_channel + b_channel * ((a_mean - b_mean) / (a_mean + b_mean))).astype(np.uint8)
    else:
        a_channel = (a_channel + a_channel * ((b_mean - a_mean)/(a_mean + b_mean))).astype(np.uint8)

    ## Combine channel
    Result = cv2.merge([l_channel_processed, a_channel, b_channel])
    Result = cv2.cvtColor(Result, cv2.COLOR_LAB2BGR)
    return Result


