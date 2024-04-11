import cv2
import numpy as np

def process_block(block, lc_variance, block_mean, block_variance, beta):
    alpha = ((lc_variance) / block_variance)
    if alpha < beta:
        block = block_mean + (alpha * (block - block_mean))
    else:
        block = block_mean + (beta * (block - block_mean))

    return block

def LACE(input_img: np.ndarray, beta: float):
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
