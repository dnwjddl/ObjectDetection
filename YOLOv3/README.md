- ```DarkNet.py``` : DarkNet Layer 제작
- ```YOLOv3_architecture.py``` : Layer Model 제작

#### cfg 파일 (Configuration file)

- cfg 파일은 블록단위로 네트워크의 Layout을 나타냄

저자가 제공하는 공식 cfg 파일 사용  

```python
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

Configuration 파일을 열면

```python
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

