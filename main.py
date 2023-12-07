import os
import glob
import argparse
import cv2
from ultralytics import YOLO
from MLLE.LACC import LACC
from MLLE.LACE import LACE

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', type=str, default='img', choices=['image', 'img', 'video', 'vid'])
parser.add_argument('-d', '--detect', type=bool, default=False)
parser.add_argument('-b', '--beta', type=float, default=1.5)
args = parser.parse_args()

def run():
    folder_path = './Input'
    output_path = './Output'
    model = YOLO('./model/urchin.pt')  ## Change model here ###
    
    ## img
    if args.type in ['img', 'image']:
        image_files = glob.glob(os.path.join(folder_path, '*.png')) + \
                    glob.glob(os.path.join(folder_path, '*.jpg'))
        for img_path in image_files:
            img = cv2.imread(img_path)/255.
            result, _ = LACC(img)
            result = LACE(result*255, args.beta)
            if args.detect:
                result = model(source=result, conf=0.6)
                result = result[0].plot()
            cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)), result)
            print(f'Image {img_path} Done!!!')
    
    ## video
    elif args.type in ['video', 'vid']:
        files = glob.glob(os.path.join(folder_path, '*.mp4'))
        for path in files:
            is_run = False
            cap = cv2.VideoCapture(path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(os.path.join(output_path, os.path.basename(path)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            while True:
                ret, frame = cap.read()
                if not ret : break
                result, is_run = LACC(frame/255., is_vid=True, is_run=is_run)
                result = LACE(result*255, beta=args.beta)
                if args.detect:
                    result = model(source=result, conf=0.5, verbose=False)
                    result = result[0].plot()
                out.write(result)
                print(f'Video ({cap.get(1)}/{cap.get(7)}) processed')
            cap.release()
            out.release()

if __name__ == '__main__':
    run()
    