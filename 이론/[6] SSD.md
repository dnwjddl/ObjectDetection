## SSD (The Single Shot Detector)

> **아웃풋을 만드는 공간을 나눈다. (multi-feature map). 각 feature map에서 다른 비율과 스케일로 Default box를 생성하고 모델을 통해 계산된 좌표와 클래스 값에 default box를 활용해 최종 bounding box 를 생성**

![image](https://user-images.githubusercontent.com/72767245/108738210-b8166c00-7576-11eb-9bde-38330ef9cfc9.png)


- Yolo의 문제점은 입력 이미지를 7x7 크기의 그리드로 나누고, 각 그리드 별로 Bounding Box Prediction을 진행하기 때문에 그리드 크기보다 작은 물체를 잡아내지 못하는 문제
- 신경망을 모두 통과하면서 컨볼루션과 풀링을 거쳐 coarse 한 정보만 남은 마지막 단 피쳐맵만 사용하기 때문에 정확도가 하락함

### Architecture

![image](https://user-images.githubusercontent.com/72767245/108737658-2c9cdb00-7576-11eb-9037-25665ea53338.png)

- SSD는 300x300 크기의 이미지를 입력받아 ImageNet으로 pretrained된 VGG의 Conv5_3층까지 통과하며 Feature 추출
  - VGG-16 모델을 가지고 와서 conv4_3까지 적용하는 것을 baseNetwork로 두고 처리 (38x38x512)

![image](https://user-images.githubusercontent.com/72767245/108805339-34df2f80-75e3-11eb-958e-9d66862f94ca.png)

**multi feature maps**: 38x38, 19x19, 10x10, 3x3, 1x1 <br><br>


- 추출된 Feature map을 convolution을 거쳐 그 다음 층에 넘겨주는 동시에 Object Detection 수행
  - 각 feature map을 convolution 연산을 거치면 우리가 에측하고자 하는 bounding box의 class 점수와 offset을 얻음
  - convolution filter size는 3x3x(#바운딩 박스갯수 x(class score + offset))
  - 각 feautre map 당 다른 스케일을 적용해 default 박스 간의 IoU을 계산한다음 미리 0.5 이상이 되는 box들만 1로 고려대상에 포함 나머지는 0
    - 그림처럼 3개의 feature map에서만 box가 detect
  - NMS를 통해 최종 Detect



![image](https://user-images.githubusercontent.com/72767245/108805993-d8304480-75e3-11eb-997d-e5386159dab0.png)


- 이전 Fully Convolution Network에서 컨볼루션을 거치면서 디테일한 정보들이 사라지는 문제점을 앞단의 feature map들을 끌어오는 방식으로 해결

---

#### [1] Multi-scale Feature maps for detection
- 38x38, 19x19, 10x10, 5x5, 3x3, 1x1 의 feature map들을 의미
- yolo는 7x7 grid 하나 뿐이지만 SSD는 전체 이미지를 여러 grid size로 나누고 output과 연겨 
- 큰 feature map에서는 작은 물체 탐지, 작은 feature map에서는 큰 물체 탐지

---

#### [2] Convolutional predictiors for detection
- 이미지부터 최종 feature map까지는 Conv(3x3, s=2)로 연결
- output과 연결된 feature map은 3x3xp 사이즈의 filter로 컨볼루션 연산 (yolo v1에서는 output 과 fully connected)
- 예측된 output은 class, cateory 점수와 default box에 대응되는 offset을 구함

---

#### [3] Default boxes and aspect ratio
```default bounding box```라는 것을 만들고 그 default box와 대응되는 자리에서 예측되는 박스의 offset과 per class scores를 예측
- 6개의 피쳐맵(마지막 6개의 피쳐맵, Output과 직결된)은 각각 Conv(3x3x(#bb x (c + offset))) 연산을 통해 Output 형성
- Ouput은 각 셀 당 ##bb개의 바운딩 박스를 예측

![image](https://user-images.githubusercontent.com/72767245/108809847-0f572380-75ed-11eb-8499-540ec6359e3d.png)


---

```Ground Truth Box``` : 우리가 예측해야 하는 정답 박스 <br>
```Predicted Box``` : Extra Network의 5x5의 feature map에서 output(predicted box)을 위해 conv 연산을 하면 총 5x5x(6x(21+4))값 형성(= grid cell x grid cell x (# of bb) x (class + offset)) <br>
```Default Box``` : 5x5 feature map은 각 셀당 6개의 default box를 가지고 있음 <br>

- default box의 w, h는 feature map의 scale에 따라 서로 다른 s 값과 서로 다른 aspect ratio인 a 값을 이용해 도출
- default box의 cx와 cy는 feature map size와 index에 따라 결정
<br>
default box와 ground Truth Box 간의 IOU를 계산하여 0.5이상의 값들은 1(positive), 아닌 값들은 0으로 할당.
> 예를 들어, 그림과 같이 5x5의 feature map의 13번째 셀(가운데)에서 총 6개의 default box와 predicted bounding box가 있는데, 같은 순서로 매칭되어 loss를 계산한다. 매칭된(x=1, positive) default box와 같은 순서의 predicted bounding box에 대해서만 offset 에 대한 loss를 고려한다.


---

#### Feature Map

![image](https://user-images.githubusercontent.com/72767245/108804762-81c20680-75e1-11eb-82d6-c7138fad08b6.png)

VGG를 통과하며 얻은 Feature map을 대상으로 컨볼루션을 계속 진행하여 최종적으로는 1x1 크기의 Feature map까지 뽑아냄  
각 단계 별로 추출된 Feature map은 Detector & Classifier를 통과시켜 Object Detection수행

---

#### Detector & Classifier

![image](https://user-images.githubusercontent.com/72767245/108804860-cea5dd00-75e1-11eb-8e28-082649f9c8ef.png)

하나의 그리드마다 크기가 각기 다른 Default Box들을 계산 
<br>
(```Default Box```: Faster R-CNN에서 anchor의 개념으로 비율과 크기가 각기 다른 기본 박스들을 먼저 설정해두어 Bounding box를 추론하는데 도움이 되는 장치)  
<br>
**Default Box**  
![image](https://user-images.githubusercontent.com/72767245/108804984-2d6b5680-75e2-11eb-8be3-b7e6ece81c9e.png)

- 고양이는 작은 물체, 강아지는 큰 물체
- 높은 해상도의 feature map에서는 작은 물체를 잘 잡아내고, 낮은 해상도에서는 큰물체를 잘 잡아냄
- 각각의 Feature map을 가져와서 비율과 크기가 각기 다른 Default Box를 투영함
- 이렇게 찾아낸 박스들에 ```Bounding box regression```을 적용하고 ```Confidence level```을 계산
- **YOLO**에서는 아무런 기본 값 없이 2개의 box를 예측하겠금 한다
<br>
**Convolution** <br>
- feature map에 3x3 컨볼루션을 적용하여(padding = 1, 크기 보존) bounding box regression 값을 계산
- default box들의 x, y, w, h의 조절 값을 나타내므로 4차원 벡터
- 인덱스 하나에 3개의 default box를 적용하였으므로 결과 feature map의 크기는 5x5x12 <br>
- 
**Classification**<br>
- 각각의 default box마다 모든 클래스에 대하여 Classification 진행
- 총 20개의 class +1 (배경 클래스) x default box 수이므로 최종 feature결과의 크기는 5x5x63 이 된다.
- 1 Step end-to-end Object Detection

---

#### Generate Default Box 

 
![image](https://user-images.githubusercontent.com/72767245/108811131-06b41c80-75f0-11eb-9f6d-dd58f4b2bf50.png)


---
### Training Objective

전체 로스는 각 클래스 별로 예측한 값과 실제 값 사이의 차인 Lconf와 바운딩 박스 리그레션 예측 값과 실제 값 사이의 차인 Lloc를 더한 값<br>

![image](https://user-images.githubusercontent.com/72767245/108811197-2b0ff900-75f0-11eb-9c1e-be858919d8ce.png)

**Lconf**<br>
- cross entropy Loss
- 모델이 물체가 있다고 판별한 default box들 가운데서 해당 박스의 ground truth 박스하고만 cross entropy loss를 구한다
- 물체가 없다고 판별한 default box들 중에 물체가 있을 경우의 loss를 계산

![image](https://user-images.githubusercontent.com/72767245/108811223-37945180-75f0-11eb-92e3-db9af1060b4c.png)

**Lloc**<br>
- smoothL1은 Robust bounding box regression loss와 같음
- bounding box regression시에 사용하는 예측값들
-  x, y 좌표 값은 절대 값이기 때문에 예측값과 실제 값 사이의 차를 default 박스의 너비 혹은 높이로 나눔 (0과 1사이로 정규화 가능)
  - 너비와 높이의 경우엔 로그를 씌워서 정규화 시킨 것  

![image](https://user-images.githubusercontent.com/72767245/108811406-93f77100-75f0-11eb-8fb4-45ba0100dcab.png)

---

- low level의 Object일 경우, layer을 충분히 거치지 못한 feature이기 때문에 좋은 결과가 안나옴
- Data Augmentation을 통해서 해결을 함

---

### 결과
- 속도, 정확도 측면에서 성능 SOTA 가 된 이유는
  - Output layer 와 FC 하지 않고 Conv를 이용(Weight 수 급감, 속도 증가)
  - 여러 Feature map은 한 이미지를 다양한 grid로 접근하고 다양한 크기의 물체들을 detect 할 수 있게 함
  - default boc 사용은 weight initialize와 normalize 효과를 동시에 가져 올 수 있을 듯
  - 6개의 bounding box를 통해 겹치는 좌표의 다양한 물체 detect 가능

