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


## Network

