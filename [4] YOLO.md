# YOLO(You Only Look Once)
**"You Only Look Once: Unified, Real-Time Object Detection (2016)"**

- ```YOLO```는 지도학습
- Pascal VOC dataset 사용 | 물체에 대한 클래스(C), 위치 정보(X,Y(기준점은 물체의 정중앙을 말함),Width, Height)

## YOLO의 세가지 특징
### You Only Look Once: 이미지 전체를 단 한번만 본다  
R-CNN은 이미지에서 일정한 규칙으로 이미지를 여러장 쪼개서 CNN 모델을 통과 시키기 때문에 한장의 이미지에서 객체 탐지를 수행해도 실제로 수천장의 모델을 통과 시키게 됨  
```YOLO```는 이미지 전체를 말 그대로 단 한번만 본다

### Unified: 통합된 모델을 사용
다른 객체 탐지 모델들은 다양항 전처리 모델과 인공신경망을 결합하여 사용  
```YOLO```는 단하나의 인공신경망에서 이를 전부 처리 (1step Object Detection)

### Real-Time Object Detection: 실시간 객체 탐지
```YOLO```가 성능이 좋은 객체 탐지 모델은 아니지만, 실시간으로 여러장의 이미지를 탐지할 수 있다  
Fast R-CNN이 0.5fps의 성능을 가진 반면 ```YOLO```는 45fps의 성능을 가짐  
이는 영상을 스트리밍 하면서 동시에 화면 상의 물체를 부드럽게 구분할 수 있다.  

-------------------------------
기존에 ```1) region proposal```, ```2)classification``` 두 단계로 나누어서 진행  
region proposal 단계를 제거하고 한번에 Object Detection을 수행하는 구조를 갖는다  

## Unified Detection
  <p align="center"><img src="https://user-images.githubusercontent.com/72767245/103547275-8a526680-4ee7-11eb-918a-44e48e2d38fd.png" width = 70%></p>

1️⃣) 입력 이미지를 S X S 그리드 영역으로 나눔(실제 입력 이미지 아님)  
2️⃣1️⃣) 각 그리드 영역에서 먼저 물체가 있을 만한 영역에 해당하는 B개의 **Bounding Box**를 예측  
  - (x,y,w,h)로 나타내어 지는데 (x, y)는 bounding box의 중심점 좌표. w,h는 넓이와 높이
  - 해당 박스의 신뢰도를 나타내는 Confidence를 계산(cell에 object가 존재하지 않는다면 confidence score은 0이 됨)
  
해당 그리드에 물체가 있을 확률 ```Pr(Object)``` 와 예측한 박스와 Ground Truth 박스와의 겹치는 영역을 비율로 나타내는 ```IoU``` 곱하여 계산  
  <p align="center"><img src="https://user-images.githubusercontent.com/72767245/103531915-96c9c580-4ecd-11eb-949f-e85dc95b8529.png" width = 30%></p>
    
✔ **Bounding Box**
- 바운딩 박스 B는 X, Y좌표, 가로, 세로 크기 정보와 Confidence Score 수치를 가지고 있음 
- Score는 B가 물체를 영역으로 잡고 있는지와 클래스를 잘 예측하였는지를 나타냄

2️⃣2️⃣) 각각의 그리드마다 C개의 클래스에 대하여 해당 클래스일 확률(**Conditional Class Probability**)을 계산  
(기존의 Object Detection은 클래스 수 + 1(배경)을 넣어 Classification하지만 yolo는 불가능)  
<br>

<p align="center"><img src="https://user-images.githubusercontent.com/72767245/103534761-dd6dee80-4ed2-11eb-84df-c8300b273326.png" width = 30%></p>

- 예) 만약 x가 grid cell의 가장 왼쪽에 있다면 x=0, y가 grid cell 중간에 있다면 y=0.5
- 예) bbox의 width가 이미지 width의 절반이라면 w=0.5  

✔ **Class Probability**
  - 그리드 셀 안에 있는 그림의 분류 확률

3️⃣) 최종적으로 클래스 조건부 확률 C와 각 바운딩 박스의 Confidence예측값을 곱하면 각 박스의 클래스 별 Confidence Score의 수치를 구할 수 있음

![image](https://user-images.githubusercontent.com/72767245/103536453-e3190380-4ed5-11eb-9ae1-c5c06579f880.png)


## Network [Pre-trained Network, Training Network, Reduction Layer]

전체 네트워크 디자인과 손실함수에 대한 소개
![image](https://user-images.githubusercontent.com/72767245/103548557-511af600-4ee9-11eb-879b-721bb3881c4d.png)

**24개의 Convolutional Layer(Conv Layer)과 2개의 Fully-Connected Layer(FC layer)**

### Pre-trained Network
GoogLeNet을 이용하여 ImageNet 1000 class dataset을 사전에 학습한 결과를 Fine-Tuning한 네트워크
이 네트워크는 20개의 Conv Layer로 구성

88%의 정확도를 사전에 학습하였다

본래 ImageNet의 데이터 셋은 224x224의 크기를 가진 이미지 데이터이지만, 어째서인지 객체 감지를 학습할 때는 선명한 이미지보다는 경계선이 흐릿한 이미지가 더 학습이 잘된다고 하여 Scale Factor를 2로 설정하여 이미지를 키워 448x448x3이 이미지를 입력 데이터로 받음


### Reduction Layer
Conv Layer를 통과할때 사용하는 Filter연산이 수행시간을 많이 잡아 먹기 때문에 무작정 네트워크를 깊게 쌓기에는 부담  
이를 해결하기 위하여 ResNet과 GoogLeNet등의 기법이 제안됨. GoogLeNet의 기법을 응용하여 연산량은 감소하면서 층은 깊게 쌓는 방식을 이용함


### Training Network
Pre-trained Network에서 학습한 feature를 이용하여 Class Probability와 Bounding box를 학습하고 예측하는 네트워크

### Loss Function (3가지 원칙)
❔ 이미지를 분류하는 classifier 문제를 bounding box를 만드는 regression문제로 생각  
❕ **Sum-Squared Error(SSD) 이용**  
❔ 바운딩 박스를 잘 그렸는지 평가하는 Localization Error와 박스안의 물체를 잘 분류했는지 평가하는 Classification Error의 패널티를 다르게 평가. 특히 박스안의 물체가 없는 경우에는 Confidence Score를 0으로 만들기 위해 Localication Error에 더 높은 패널티를 부과  
❕ λ_coord 과 λ_noobj 두개의 변수를 이용 (본 논문에서는 λ_coord = 5, λ_noobj = 0.5로 설정)  
❔ 많은 바운딩 박스 중에 IoU수치가 가장 높게 생성된 바운딩 박스만 학습에 참여. 이는 바운딩 박스를 잘 만드는 셀은 더욱 학습을 잘하도록 높은 Confidence Score를 주고 나머지 셀은 바운딩 박스를 잘 만들지 못하더라도 나중에 Non-max suppression을 통해 최적화하기 위함

|변수| 설명 |
|:----:|:----------:|
|S|그리드 셀의 크기를 의미. 행렬이기 때문에 전체 그리드 셀의 크기는 S^2|
|B|S_i 셀의 바운딩 박스를 의미|
|x,y,w,h|바운딩 박스의 좌표 및 크기를 의미|
|C|각 그리드 셀이 구분한 클래스와 같음|
|1번|5로 설정된 λ_coord 변수로서 Localization에러에 5배 더 높은 패널티를 부여하기 위해서 사용|
|2번|if문과 동일한 역할. i번째 셀의 j번 바운딩 박스만을 학습하겠다는 의미로 사용. 하지만 모든 셀에 대해서 바운딩 박스 학습이 일어나지 않고 각 객체마다 IoU가 가장 높은 바운딩 박스인 경우에만 패널티를 부과해서 학습을 더 잘하도록 유도|


![image](https://user-images.githubusercontent.com/72767245/103561553-0c995580-4efd-11eb-8238-386b5b5701b8.png)



