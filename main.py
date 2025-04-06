import os
import glob
import argparse
import cv2
from ultralytics import YOLO
from utils.LACC import LACC
from utils.LACE import LACE
from utils.fusion import fusion

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', type=str, default='img', choices=['image', 'img', 'video', 'vid'])
parser.add_argument('-m', '--mode', choices=['mlle', 'fusion'])
parser.add_argument('-d', '--detect', action='store_true')
parser.add_argument('-b', '--beta', type=float, default=1.5)
args = parser.parse_args()

def run():
    if not (args.detect or args.mode): return 
    
    folder_path = './Input'
    output_path = './Output'
    if not args.detect:
        model = YOLO('./model/turtle.pt')  ## CHANGE MODEL PATH HERE ##

    # img
    if args.type in ['img', 'image']:
        image_files = glob.glob(os.path.join(folder_path, '*.png')) + \
                    glob.glob(os.path.join(folder_path, '*.jpg'))
        for img_path in image_files:
            img = cv2.imread(img_path)
            if args.mode == 'mlle':
                img, _ = LACC(img/255.)
                img = LACE(img*255, args.beta)
            elif args.mode == 'fusion':
                img, _ = LACC(img/255.)
                img = fusion(img)*255
            if args.detect:
                img = model(source=img, conf=0.6)
                img = img[0].plot()
            cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)
            print(f'Image {img_path} Done!!!')

    # video
    elif args.type in ['video', 'vid']:
        files = glob.glob(os.path.join(folder_path, '*.mp4'))
        for path in files:
            is_run = False
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(os.path.join(output_path, os.path.basename(path)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            all_frame = cap.get(7)
            while True:
                ret, frame = cap.read()
                if not ret : break
                if args.mode == 'mlle':
                    frame, is_run = LACC(frame/255., is_vid=True, is_run=is_run)
                    frame = LACE(frame*255, beta=args.beta)
                if args.mode == 'fusion':
                    frame, is_run = LACC(frame/255., is_vid=True, is_run=is_run)
                    frame = fusion(frame)*255
                if args.detect:
                    frame = model(source=frame, conf=0.5, verbose=False)
                    frame = frame[0].plot()
                out.write(frame)
                print(f'Video ({cap.get(1)}/{all_frame}) {path} processed')
            cap.release()
            out.release()

if __name__ == '__main__':
    run()