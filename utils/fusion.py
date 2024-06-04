import cv2
import numpy as np

def fusion(input_img):
    '''
    input[0, 1]
    output[0, 1]
    '''
    sigma = 20
    ksize = 4 * sigma + 1
    
    ## blur
    Igauss_blurred = cv2.GaussianBlur(input_img, (ksize, ksize), 20)
    
    ## Norm
    gain = 0.3
    Norm = (input_img - gain * Igauss_blurred)

    ## Histogram Equalization
    for n in range(3):
        Norm[:, :, n] = cv2.equalizeHist((Norm[:, :, n] * 255).astype(np.uint8)) / 255.0

    ## Sharp
    I_sharp = (input_img + Norm) / 2

    ## Gamma correct 
    gamma = 1.8
    I_gamma = np.power(input_img, gamma)
    
    ## BGR to Lab
    I_sharp = I_sharp.astype(np.float32)
    I_gamma = I_sharp.astype(np.float32)
    I_sharp_lab = cv2.cvtColor(I_sharp, cv2.COLOR_BGR2Lab)
    I_gamma_lab = cv2.cvtColor(I_gamma, cv2.COLOR_BGR2Lab)


    ## Laplacian weight (W_l)

    # For I_sharp
    R1 = I_sharp_lab[:, :, 0] / 255.0

    # For I_gamma
    R2 = I_gamma_lab[:, :, 0] / 255.0
    
    ## Saliency weight (W_S)
    W_S1 = saliency_detection(I_sharp)
    W_S1 = W_S1 / np.max(W_S1)

    W_S2 = saliency_detection(I_gamma)
    W_S2 = W_S2 / np.max(W_S2)

    W_C1 = np.abs(cv2.Laplacian(R1, cv2.CV_32F))
    W_C2 = np.abs(cv2.Laplacian(R2, cv2.CV_32F))

    ## Saturation weight (W_Sat)
    W_SAT1 = np.sqrt(1/3 * ((I_sharp[:,:,0] - R1)**2 + 
                            (I_sharp[:,:,1] - R1)**2 + 
                            (I_sharp[:,:,2] - R1)**2))

    W_SAT2 = np.sqrt(1/3 * ((I_gamma[:,:,0] - R2)**2 + 
                            (I_gamma[:,:,1] - R2)**2 + 
                            (I_gamma[:,:,2] - R2)**2))

    ## Normalized weight
    W1 = (W_C1 + W_S1 + W_SAT1 + 0.1) / (W_C1 + W_S1 + W_SAT1 + W_C2 + W_S2 + W_SAT2 + 0.2)
    W2 = (W_C2 + W_S2 + W_SAT2 + 0.1) / (W_C1 + W_S1 + W_SAT1 + W_C2 + W_S2 + W_SAT2 + 0.2)

    ## gaussian pyramid
    level = 8
    Weight_1 = gaussian_pyramid(W1, level)
    Weight_2 = gaussian_pyramid(W2, level)

    ## laplacian pyramid
    B1 = laplacian_pyramid(I_sharp[:, :, 0], level)
    G1 = laplacian_pyramid(I_sharp[:, :, 1], level)
    R1 = laplacian_pyramid(I_sharp[:, :, 2], level)

    ## gamma img
    B2 = laplacian_pyramid(I_gamma[:, :, 0], level)
    G2 = laplacian_pyramid(I_gamma[:, :, 1], level)
    R2 = laplacian_pyramid(I_gamma[:, :, 2], level)

    ## fusion
    Rr = []
    Rg = []
    Rb = []

    for k in range(level):
        Rr.append(Weight_1[k] * R1[k] + Weight_2[k] * R2[k])
        Rg.append(Weight_1[k] * G1[k] + Weight_2[k] * G2[k])
        Rb.append(Weight_1[k] * B1[k] + Weight_2[k] * B2[k])

    B = np.clip(pyramid_reconstruct(Rb), 0, 1)
    G = np.clip(pyramid_reconstruct(Rg), 0, 1)
    R = np.clip(pyramid_reconstruct(Rr), 0, 1)
    
    return cv2.merge([B, G, R])

def saliency_detection(img):
    # Gaussian blur
    gfrgb = cv2.GaussianBlur(img, (3,3), 0)
    
    # Convert image from BGR to Lab color space
    lab = cv2.cvtColor(gfrgb, cv2.COLOR_BGR2Lab).astype(np.float64)
    
    # Compute Lab average values
    l, a, b = cv2.split(lab)
    lm, am, bm = np.mean(l), np.mean(a), np.mean(b)
    
    # Compute the saliency map
    sm = (l-lm)**2 + (a-am)**2 + (b-bm)**2
    
    return sm

def gaussian_pyramid(img, level):
    h = np.array([1, 4, 6, 4, 1]) / 16
    filt = np.outer(h, h)
    out = []
    
    filtered_img = cv2.filter2D(img, -1, filt, borderType=cv2.BORDER_REPLICATE)
    out.append(filtered_img)
    
    temp_img = filtered_img
    for i in range(1, level):
        temp_img = temp_img[::2, ::2]  # Downsample by a factor of 2
        filtered_img = cv2.filter2D(temp_img, -1, filt, borderType=cv2.BORDER_REPLICATE)
        out.append(filtered_img)
        
    return out

def laplacian_pyramid(img, level):
    out = [img]
    temp_img = img
    
    for i in range(1, level):
        temp_img = temp_img[::2, ::2]  # Downsample by a factor of 2
        out.append(temp_img)
    
    for i in range(level - 1):
        m, n = out[i].shape
        upscaled = cv2.resize(out[i+1], (n, m), interpolation=cv2.INTER_LINEAR)
        out[i] = out[i] - upscaled
        
    return out

def pyramid_reconstruct(pyramid):
    level = len(pyramid)
    for i in range(level - 1, 0, -1):
        m, n = pyramid[i-1].shape[:2]
        upscaled = cv2.resize(pyramid[i], (n, m), interpolation=cv2.INTER_LINEAR)
        pyramid[i-1] += upscaled
        
    return pyramid[0]