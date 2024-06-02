# Underwater image color correction and adaptive contrast enhancemention and yolo detect python
The original uses matlab and it p-code.

I used python to restore more than 80% of the algorithms in the paper.

And i add a method to exchange wrong channels problem.

For contrast enhancement, this repo have fusion method that can replace LACE.

This repo have webpage for apply MLLE and fusion. Webpage can upload image and download image after process.

This repo Webpage: [MLLE-webpage](https://nomi30701.github.io/Underwater-image-color-correction-adaptive-contrast-enhancemention-and-yolo-detect-python/)

* Paper:
[Underwater Image Enhancement via Minimal Color Loss and Locally Adaptive Contrast Enhancement](https://ieeexplore.ieee.org/document/9788535).

* Paper:
[Color Balance and Fusion for Underwater Image Enhancement](https://ieeexplore.ieee.org/document/8058463)

* Source matlab code:
[MLLE](https://github.com/Li-Chongyi/MMLE_code)

* Test image from:
[UIEB dataset](https://li-chongyi.github.io/proj_benchmark.html)

# Preview

[Preview Video](https://www.youtube.com/watch?v=DxaS2R58Tyg)

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/webapp_preview.jpg" height=70% width=70% />

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/38.png_reslut.jpg" height=70% width=70% />

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/906_img_.png_reslut.jpg" height=70% width=70% />

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/91_img_.png_reslut.jpg" height=70% width=70% />

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/image%20(2).png_reslut.jpg" height=70% width=70% />

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/image%20(7).png_reslut.jpg" height=70% width=70% />

<img src="https://github.com/nomi3070/Underwater-image-correct-and-adaptive-contrast-enhancemention-python/blob/main/Preview%20image/image%20(8).png_reslut.jpg" height=70% width=70% />

# Requirements
```
pip install -r requirements.txt
```
> If you want to use gpu for object detection, please install **gpu version** [pytorch](https://pytorch.org/).

# How to use
Put the image(.jpg .png) or .mp4 in the `Input` folder.

This repo provides models for detecting sea turtles and sea urchins.

If you want to change the model please open the `main.py` file.

`model = YOLO('to/your/model/path')`

# Usage example
**Using the command line interface (CLI)**

MLLE for image
```
python main.py --mlle
```

or

```
python main.py -m
```

Change to process video, detect object by yolo, beta

```
python main.py --type video --mlle --detect --beta 2
```

or

```
python main.py -t vid -m -d -b 2
```

parameters: 
   - --type or -t ['img', 'image', 'vid', 'video']

   - --mlle
   - -m

   - --detect
   - -d

   - --beta [int, float]
   - -b [int, float]
