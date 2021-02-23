## SSD (The Single Shot Detector)

> **아웃풋을 만드는 공간을 나눈다. (multi-feature map). 각 feature map에서 다른 비율과 스케일로 Default box를 생성하고 모델을 통해 계산된 좌표와 클래스 값에 default box를 활용해 최종 bounding box 를 생성**

![image](https://user-images.githubusercontent.com/72767245/108738210-b8166c00-7576-11eb-9bde-38330ef9cfc9.png)


- Yolo의 문제점은 입력 이미지를 7x7 크기의 그리드로 나누고, 각 그리드 별로 Bounding Box Prediction을 진행하기 때문에 그리드 크기보다 작은 물체를 잡아내지 못하는 문제
- 신경망을 모두 통과하면서 컨볼루션과 풀링을 거쳐 coarse 한 정보만 남은 마지막 단 피쳐맵만 사용하기 때문에 정확도가 하락함

### Multi Scale Feature Maps for Detection

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

#### Multi-scale Feature maps for detection
- 38x38, 19x19, 10x10, 5x5, 3x3, 1x1 의 feature map들을 의미
- yolo는 7x7 grid 하나 뿐이지만 SSD는 전체 이미지를 여러 grid size로 나누고 output과 연겨 
- 큰 feature map에서는 작은 물체 탐지, 작은 feature map에서는 큰 물체 탐지

---

#### Convolutional predictiors for detection
- 이미지부터 최종 feature map까지는 Conv(3x3, s=2)로 연결
- output과 연결된 feature map은 3x3xp 사이즈의 filter로 컨볼루션 연산
- 예측된 output은 class, cateory 점수와 default box에 대응되는 offset을 구함

---

#### Feature Map

![image](https://user-images.githubusercontent.com/72767245/108804762-81c20680-75e1-11eb-82d6-c7138fad08b6.png)

VGG를 통과하며 얻은 Feature map을 대상으로 컨볼루션을 계속 진행하여 최종적으로는 1x1 크기의 Feature map까지 뽑아냄  
각 단계 별로 추출된 Feature map은 Detector & Classifier를 통과시켜 Object Detection수행

---

#### Detector & Classifier

![image](https://user-images.githubusercontent.com/72767245/108804860-cea5dd00-75e1-11eb-8e28-082649f9c8ef.png)

하나의 그리드마다 크기가 각기 다른 Default Box들을 계산 (Default Box: Faster R-CNN에서 anchor의 개념으로 비율과 크기가 각기 다른 기본 박스들을 먼저 설정해두어 Bounding box를 추론하는데 도움이 되는 장치)  

**Default Box**  
![image](https://user-images.githubusercontent.com/72767245/108804984-2d6b5680-75e2-11eb-8be3-b7e6ece81c9e.png)

- 고양이는 작은 물체, 강아지는 큰 물체
- 높은 해상도의 feature map에서는 작은 물체를 잘 잡아내고, 낮은 해상도에서는 큰물체를 잘 잡아냄
- 각각의 Feature map을 가져와서 비율과 크기가 각기 다른 Default Box를 투영함
- 이렇게 찾아낸 박스들에 Bounding box regression을 적용하고 Confidence level을 계산
- **YOLO**에서는 아무런 기본 값 없이 2개의 box를 예측하겠금 한다

