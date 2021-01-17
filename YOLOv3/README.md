- ```Darknet53.py``` : DarkNet-53 Layer 제작
- ```[TEST] yolov3_architecture.py``` : Layer Model 제작

#### cfg 파일 (Configuration file)

- cfg 파일은 블록단위로 네트워크의 Layout을 나타냄

저자가 제공하는 공식 cfg 파일 사용  

```python
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

**Configuration** 파일을 열면

```c
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
```
저자의 cfg 파일을 토대로 ```yolov3_architecture.py``` 작성

```c
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
```
