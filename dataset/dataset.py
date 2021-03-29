import cv2
import numpy as np
from tensorflow.io import gfile
from multiprocessing import Process, Queue

def transform(img, resize_height=48, resize_width=48, resize=True):
    if resize:
        resized_img = cv2.resize(img, (resize_width, resize_height))
    else:
        resized_img = img
    return np.array(resized_img, dtype=np.float32)/127.5 - 1

def inverse_transform(img):
    return(img+1)/2

def dataset_gen():
    videos = gfile.glob("./results/*view0.mp4")
    idata = [[] for _ in range(25)]

    for itr, name in enumerate(videos):
        length = 0
        cap = cv2.VideoCapture(name)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame, 48, 48)
                if length%2 == 1:
                    frames.append(frame)
                length += 1
            else:
                break
        cap.release()

        if itr % 100 == 0:
            print(itr, "/", len(videos))

        for i, frame in enumerate(frames):
            idata[i].append(frame)

    vdata = np.array(idata)
    np.save('singleview_push_sim_'+str(len(videos)), vdata)
    return vdata

if __name__ == "__main__":
    dataset_gen()