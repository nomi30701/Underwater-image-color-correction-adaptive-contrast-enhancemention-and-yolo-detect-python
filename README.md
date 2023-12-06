# Underwater image correct and adaptive contrast enhancemention python
The code creator only uses matlab and is p-code.
I use python to restore more than 80% of the algorithms in this paper.
And I added a method to exchange channels.

* Paper:
[Underwater Image Enhancement via Minimal Color Loss and Locally Adaptive Contrast Enhancement](https://ieeexplore.ieee.org/document/9788535).

* Source matlab code:
[MLLE](https://github.com/Li-Chongyi/MMLE_code)

* Test image from:
[UIEB dataset](https://li-chongyi.github.io/proj_benchmark.html)
# Preview
![img 1](https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/38.png_reslut.jpg)

![img 2](https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/906_img_.png_reslut.jpg)

![img 3](https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/91_img_.png_reslut.jpg)

# How to run the code
* Requirements:
```
pip install -r requirements.txt
```

* Run the code (For image, no detect object):
```
python main.py
```
![img 1](https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Input/906_img_.png)

* Change to video, detect object, parameter beta 

Default: --type img --detect False --beta 1.5
```
python main.py --type video --detect True --beta 2
```
