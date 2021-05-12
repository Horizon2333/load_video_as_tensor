# -*- coding: utf-8 -*-
"""
@Author : Horizon
@Date   : 2021-03-20 00:30:23
"""

# This code is used to test the time cost to load a video and transform it into PILImage.

# Methods to be tested
import cv2
from PIL import Image
import skvideo.io
import kornia
import skimage
import imageio
import torchvision
from vidgear.gears import CamGear
import av

# Other library
import torch
from torchvision import transforms

import numpy as np

import os
import time

#from rich.console import Console
#from rich.table import Table

def cv2_loadvideo(video_path):

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()

    frames_tensor = []

    while(True):
        
        ret, frame = cap.read()

        # read over
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (224, 224))
        #frame = transform(frame).unsqueeze(0)

        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
    #frames_tensor = torch.cat(frames_tensor, 0)

    #print(frames_tensor.shape)

    return

def cv2_to_PIL_loadvideo(video_path):

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()

    frames_tensor = []

    transform = transforms.Compose([                                        
                                    #transforms.ToPILImage(),
                                    #transforms.Resize((224, 224)),                                    
                                    transforms.ToTensor(),
                                    ])

    while(True):
        
        ret, frame = cap.read()

        # read over
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).unsqueeze(0)

        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = torch.cat(frames_tensor, 0)

    #print(frames_tensor.shape)

    return

def sk_video_loadvideo(video_path):

    videodata = skvideo.io.vreader(video_path)

    frames_tensor = []

    for frame in videodata:

        #frame = transform(frame).unsqueeze(0)

        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
    #frames_tensor = torch.cat(frames_tensor, 0)

    #print(frames_tensor.shape)

    return

def imageio_loadvideo(video_path):

    reader = imageio.get_reader(video_path)

    frames_tensor = []

    for frame in reader:

        frame = skimage.img_as_float(frame).astype(np.float32)
        
        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)

    #print(frames_tensor.shape)

    return

def vidgear_loadvideo(video_path):

    stream = CamGear(source=video_path).start()

    frames_tensor = []

    while True:

        frame = stream.read()

        if frame is None:
            break
        
        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)

    #print(frames_tensor.shape)
    
    stream.stop()

    return

def pyav_loadvideo(video_path):

    container = av.open(video_path)

    frames_tensor = []

    # !!! This is the only difference.
    container.streams.video[0].thread_type = 'AUTO'

    for packet in container.demux():
        for frame in packet.decode():

            frame = frame.to_image()  # PIL/Pillow image
            frame = np.asarray(frame)  # numpy array
            # Do something!
            frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)

    #print(frames_tensor.shape)
    container.close()

    return

def load_frames(video_path):

    video_dir = video_path.split('.')[0]

    frame_paths = os.listdir(video_dir)

    frames_tensor = []

    for frame_path in frame_paths:
        frame = cv2.imread(os.path.join(video_dir, frame_path))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (224, 224))

        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)

    #print(frames_tensor.shape)

    return

def torchio_loadvideo(video_path):

    frames_tensor, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')

    frames_tensor = frames_tensor.div(255.0).permute((0, 3, 1, 2))

    #print(frames_tensor.shape, frames_tensor.max(), frames_tensor.min())

    return

def test(video_path, method):

    start_time = time.time()

    method(video_path)

    end_time = time.time()

    return end_time - start_time

def task(method):

    time_short = test("test_videos/short_video.avi", method)
    time_long  = test("test_videos/long_video.avi" , method)
    time_small = test("test_videos/small_video.avi", method)
    time_big   = test("test_videos/big_video.avi"  , method)

    return time_short, time_long, time_small, time_big

if(__name__ == "__main__"):

    '''

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("method"                , justify="center")
    table.add_column("initial datatype"      , justify="center")
    table.add_column("intermediate  datatype", justify="center")
    table.add_column("final datatype"        , justify="center")
    table.add_column("short video cost /s"   , justify="center")
    table.add_column("long video cost /s"    , justify="center")
    table.add_column("small video cost /s"   , justify="center")
    table.add_column("big video cost /s"     , justify="center")

    '''
        
    #1.opencv → Tensor
    time_short, time_long, time_small, time_big = task(cv2_loadvideo)
    print("opencv: ndarray → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.addrow("opencv", "ndarray", "-", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))
    
    #2.opencv → PIL → Tensor
    time_short, time_long, time_small, time_big = task(cv2_to_PIL_loadvideo)
    print("opencv: ndarray → PILImage → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("opencv", "ndarray", "PILImage", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))

    #3.scikit-video → Tensor
    time_short, time_long, time_small, time_big = task(sk_video_loadvideo)
    print("scikit-video: ndarray → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("scikit-video", "ndarray", "-", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))

    #4.imageio → Tensor
    time_short, time_long, time_small, time_big = task(imageio_loadvideo)
    print("imageio: imageio.core.util.Array → ndarray → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("imageio", "imageio.core.util.Array", "ndarray", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))
    
    #5.vidgear → Tensor
    time_short, time_long, time_small, time_big = task(vidgear_loadvideo)
    print("vidgear: ndarray → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("vidgear", "ndarray", "-", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))
    
    #6.pyav → Tensor
    time_short, time_long, time_small, time_big = task(pyav_loadvideo)
    print("pyav: av.video.frame.VideoFrame → PILImage → ndarray → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("pyav", "av.video.frame.VideoFrame", "PILImage → ndarray", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))

    #7.frames → Tensor
    time_short, time_long, time_small, time_big = task(load_frames)
    print("frames: ndarray → Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("frames", "ndarray", "-", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))
    
    #8.torchvision.io → Tensor
    time_short, time_long, time_small, time_big = task(torchio_loadvideo)
    print("torchvision.io: Tensor")
    print("{:2.2f} {:2.2f} {:2.2f} {:2.2f}".format(time_short, time_long, time_small, time_big))
    print()
    #table.add_row("torchvision.io", "Tensor", "-", "Tensor", 
    #                '{:.2f}'.format(time_short), '{:.2f}'.format(time_long), '{:.2f}'.format(time_small), '{:.2f}'.format(time_big))
    
    #console.print(table)