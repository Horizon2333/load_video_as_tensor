# -*- coding: utf-8 -*-
"""
@Author : Horizon
@Date   : 2021-03-26 17:59:29
"""

import os
import cv2

def video2jpg(video_path, output_path):

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()

    index = 0

    while(True):
        
        ret, frame = cap.read()

        # read over
        if not ret:
            break
        
        cv2.imwrite(os.path.join(output_path, "%05d.jpg" % index), frame)

        index += 1

    return

if __name__ == "__main__":

    video2jpg("test_videos/short_video.avi", "test_videos/short_video")
    video2jpg("test_videos/long_video.avi", "test_videos/long_video")
    video2jpg("test_videos/small_video.avi", "test_videos/small_video")
    video2jpg("test_videos/big_video.avi", "test_videos/big_video")