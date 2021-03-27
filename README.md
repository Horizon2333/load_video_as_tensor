# Loading videos to Tensor in PyTorch

There are many ways to load videos, but in oder to accelerate the training speed in video models, I tried different packages to see which one is the best.

The test videos is come from the [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) dataset.

## Install

1. Clone the project
```shell
git clone https://github.com/Horizon2333/load_video_as_tensor
cd load_video_as_tensor
```
2. Install dependencies
```shell
pip install -r requirements.txt
```

## Usage
1.  Transform the video to images (Because one experiment try to load frames) :

```shell
python video2jpg.py
```

2. Run the experiment:

```shell
python test_load_video_speed.py
```

## Results

|     method     |     initial datatype      | intermediate  datatype | final datatype | short video cost /s | long video cost /s | small video cost /s | big video cost /s |
| :------------: | :-----------------------: | :--------------------: | :------------: | :-----------------: | :----------------: | :-----------------: | :---------------: |
|     opencv     |          ndarray          |           -            |     Tensor     |        0.15         |        1.08        |        0.08         |       15.34       |
|     opencv     |          ndarray          |        PILImage        |     Tensor     |        0.27         |        2.27        |        0.12         |       17.35       |
|  scikit-video  |          ndarray          |           -            |     Tensor     |        0.43         |        1.32        |        0.21         |       16.00       |
|    imageio     |  imageio.core.util.Array  |        ndarray         |     Tensor     |        0.71         |        3.41        |        0.32         |       18.61       |
|    vidgear     |          ndarray          |           -            |     Tensor     |        2.21         |       66.26        |        3.44         |       25.58       |
|      pyav      | av.video.frame.VideoFrame |   PILImage → ndarray   |     Tensor     |        0.41         |        2.04        |        0.12         |       15.92       |
|     frames     |          ndarray          |           -            |     Tensor     |        0.29         |        4.73        |        0.22         |       19.83       |
| torchvision.io |          Tensor           |           -            |     Tensor     |        0.49         |        2.40        |        0.10         |       18.02       |

(Test on computer with: Intel Core i7-9750H; 16G RAM; Nvidia GeForce GTX 1660Ti with Max-Q Design)

Seems like the best choice is just to use OpenCV!



If there are something wrong with my experiment or there are better ways to load video in RAM, please tell me, thank a lot!