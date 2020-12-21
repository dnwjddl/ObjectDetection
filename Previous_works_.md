# R-CNN

## RCNN은 CNN을 본격적으로 이용하여 Object Detection에서 높은 성능을 보임
전반적인 R-CNN의 workflow
![image](https://user-images.githubusercontent.com/72767245/102717097-76253b80-4323-11eb-88bf-31440ac348ce.png)

### R-CNN의 학습과정
![image](https://user-images.githubusercontent.com/72767245/102717110-8806de80-4323-11eb-855f-b7f0e48de253.png)
1. 입력 이미지에 Selective Search 알고리즘 적용하여 이미지에 대한 후보 영역(region proposal)을 생성 (약 2000여개)
2. 각 region proposal 마다 고정된 크기(227*227) warping/crop하여 CNN 인풋으로 사용함 (*박스의 비율은 고려하지 않음)
3. 미리 이미지 넷 데이터를 통해 학습 시켜 놓은 CNN을 통과시켜 4096차원의 특징 벡터를 추출(feature map)
4-1. 추출된 벡터를 가지고 각각의 클래스(Object 종류) 마다 학습시켜놓은 SVM Classifier통과
4-2. Regressor을 통한 bounding box regression 진행(박스의 위치 조정)

![image](https://user-images.githubusercontent.com/72767245/102719610-43367400-4332-11eb-8951-c8fb02fadcbe.png)

세 단계의 multi-stage로 구성되어 있음
- Selective Search을 통한 region proposal 생성
- 각 region proposal 마다 고정된 크기의 박스 추출
- 추출된 Feature map을 사용하여 SVM Classifier과 bounding box regression 진행

#### 첫번째 stage, Region Proposal
- Region proposal, 주어진 이미지에서 물체가 있을 법한 위치를 찾는 것
- Selective Search라는 룰 베이스 알고리즘을 적용하여 2천개의 물체가 있을 법한 박스를 찾는다.
- Selective Search는 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을 법한 박스를 추론 <br>
**RCNN 이후 Region Proposal 과정은 뉴럴 네트워크가 수행하도록 발전함**

#### 두번째와 세번째 stage, Feature Extraction
- Selective Search를 통해서 찾아낸 2천개의 박스 영역은 227 x 227 크기로 리사이즈(warp)
- Image Classification으로 미리 학습되어 있는 CNN 모델을 통과하여  4096크기의 특징 벡터를 추출

- 이미지넷 데이터로 미리 학습된 CNN 모델을 가지고 온 다음, fine tune 하는 방식을 취함
- Fine tune시에는 실제 Object Detection을 적용할 데이터 셋에서 ground truth에 해당하는 이미지들을 가져와 학습시켜야함

- 각 CNN 레이어 층에서 추출된 벡터로 SVM Classifier를 학습시킴 <br>

**이미지넷으로 학습된 CNN을 가지고와서, Object Detection용 데이터 셋으로 fine tuning 한 뒤, selective search 결과로 뽑힌 이미지들로부터 특징 벡터 추출**

#### 세번째 -1 stage, Classification
CNN을 통해 추출한 벡터를 각각 클래스 별로 SVM Classifier를 학습시킴
"그냥 CNN Classifier를 쓰는 것이 SVM을 썼을 때보다 mAP성능이 낮아짐. 이는 아마도 fine tuning 과정에서 물체의 위치 정보가 유실되고 무작위로 추출된 샘플을 학습하여 발생한 것으로 보임"

#### 세번째 -1 stage, Non-Maximum Suppression - IoU
SVM을 통과하여 각각의 박스들은 어떤 물체일 확률(Score)값을 가지게 됨 <br>
가장 높은 score을 가진 박스만 남기고 나머지는 제거 <Non-Maximum Suppression> <br>

서로 다른 두 박스가 동일한 물체에 쳐져있는지 확인하는 방법(IoU: Intersection over Union) <br>
![image](https://user-images.githubusercontent.com/72767245/102718320-dc618c80-432a-11eb-892f-996788e62473.png) <br>
-> 두 박스의 교집합을 합집합으로 나눠준 값
<br>
**논문에서는 IoU가 0.5보다 크면 동일한 물체의 대상으로 판별**

  
#### 세번째 -2 stage, Bounding Box Regression(위치 교정)
물체가 있을 법한 위치를 찾고, 해당 물체의 종류를 판별할 수 있는 Classifier모델을 학습시킴. <br>

박스 표기법
<img src="https://user-images.githubusercontent.com/72767245/102718596-56dedc00-432c-11eb-8386-c3e482c9b59d.png" width="20%">

Ground Truth 에 해당하는 박스
<img src="https://user-images.githubusercontent.com/72767245/102718612-68c07f00-432c-11eb-850b-7ab333c8338c.png" width="20%"><br>

**목표: P에 해당하는 박스를 최대한 G에 가깝도록 이동시키는 함수** <br>
박스가 input으로 들어왔을 때, x,y, w,h를 각각 이동시켲는 함수들 표현<br>
<img src="https://user-images.githubusercontent.com/72767245/102718635-7fff6c80-432c-11eb-9d5f-1fb15b6a69b8.png" width="40%"><br>
x,y는 점이므로, 이미지의 크기와 상관없이 위치만 이동시켜주면 됨<br>
반면, 너비와 높이는 이미지의 크기에 비례하여 조정을 시켜주어야 함.<br>
<img src="https://user-images.githubusercontent.com/72767245/102719056-01f09500-432f-11eb-933e-02add80d63ab.png" width="50%"><br>

**학습을 통해 얻고자 하는 함수는 d함수임** <br>
φ(Pi)는 VGG넷의 pool5를 거친 피쳐맵으로, 원래의 VGG에서는 이를 쫙 펴서 4096 차원의 벡터로 만든 다음 FC에 넘겨줌. 즉, φ(Pi)를 4096 차원 벡터라고 보면 w*역시 4096 차원 벡터이다. 
<br>
<img src="https://user-images.githubusercontent.com/72767245/102718700-d79dd800-432c-11eb-9e2e-ead3bea134bb.png" width="20%"><br>
<br>
이 둘을 곱해서 구하고 싶은 값은 x, y, w, h로 이는 모두 0에서 1 사이의 값입니다. (각각을 바운딩 박스의 너비와 높이로 나누어 주므로) 즉, 0과 1 사이의 바운딩 박스 조정 값을 구하기 위해서 4096 차원의 벡터를 학습시키는 것입니다. <br>
MSE 에러함수에 L2 normalization 추가한 형태<br>
<img src="https://user-images.githubusercontent.com/72767245/102718712-e84e4e00-432c-11eb-9d80-4fb08057caae.png" width="40%"><br>
t는 P를 G로 이동시키기 위해서 필요한 이동량을 의미하며 식으로 나타내면 아래와 같다.<br>
<img src="https://user-images.githubusercontent.com/72767245/102718749-0caa2a80-432d-11eb-8d28-6c311a0f3e7b.png" width="20%"><br>

**CNN을 통과하여 추출된 벡터 x,y,w,h를 조정하는 함수의 weight를 곱해서 바운딩 박스를 조정해주는 선형회귀를 학습시키는 것**

### 학습이 일어나는 부분
- 1. 이미지넷으로 이미 학습된 부분을 가져와 fine-tuning 하는 부분
- 2. SVM Classifier를 학습시키는 부분
- 3. Bounding Box Regression

### R-CNN의 단점
Selective search에 해당하는 region proposal 만큼 CNN을 돌려야함
- **큰 저장 공간을 요구**
- **속도가 느림** <br>

기존의 CNN 아키텍처들은 모두 입력 이미지가 고정되어야 함(ex.224x224), 비율을 조정해야 했다.
- 물체의 일부분이 잘리거나, 본래의 생김새와 달라지는 문제점

R-CNN의 단점을 보완하고자 제안된 연구

# SPPNet
## Spatial Pyramid Pooling이라는 특징을 가짐
#### 각 region proposal 마다의 CNN feature map 생성(2천개의 feature map) => CNN을 적용하여 생성된 feature map을 region proposal 거침
#### 학습에서는 3배, 실제 사용시 10배-100배 속도 개선
![image](https://user-images.githubusercontent.com/72767245/102811903-7b5cb600-4409-11eb-8abf-34c7afc8eee7.png)


### 기존의 CNN 입력
* Convolution filter들은 사실 입력 이미지의 고정이 필요하지 않다.
* sliding window 방식으로 작동하기 때문에, 입력 이미지의 크기나 비율에 관계 없이 작동함 
* 입력 이미지 크기의 고정이 필요한 이유는 컨볼루션 layer 이후에 이루어지는 fully connected layer가 고정된 크기의 입력을 받기 때문 -> **Spatial Pyramid Pooling(SPP) 제안**
<br>
 
**입력 이미지의 크기 관계 없이 Conv layer을 통과시키고, FC layer 통과 전에 피쳐 맵들을 동일한 크기로 조절해주는 pooling을 적용하자**<br>
<br>
**이미지의 특징을 고스란히 간직한 feature map 얻기 가능** <br>
**사물의 크기 변화에 더 견고한 모델을 얻을 수 있음**<br>
<br>

### SPP 전체 흐름

![image](https://user-images.githubusercontent.com/72767245/102809014-a09af580-4404-11eb-9468-33692864eff6.png)

1. 전체 이미지를 미리 학습된 CNN을 통과시켜 feature map 추출 <br>
2. Selective Search를 통해서 찾은 각각의 RoI들을 제 각기 크기와 비율이 다르다. 이에 SPP를 적용하여 고정된 크기의 Feature vector를 추출<br>
3. 그 다음 FC layer 통과 -> 벡터 추출<br>
4-1. 앞서 추출한 벡터로 각 이미지 클래스 별로 binary SVM Classifier를 학습시킴<br>
4-2. 앞서 추출한 벡터로 bounding box regressor 학습 시킴<br>

![그림](https://user-images.githubusercontent.com/72767245/102808966-882adb00-4404-11eb-8fb8-3e80fef356f8.png)
##### CNN: Conv1 -> Conv2 -> Conv3 -> Conv4 -> Conv5 -> SPP -> FC6 -> FC 7 -> Softmax

#### R-CNN 방법과 SPPNet 방법 차이
- R-CNN은 Image 영역(Selective Search 방법으로 수천개의 box)을 추출하고 warp 시킨다
  - 원하지 않는 왜곡
  - 이후 추출한 영역(2천개)를 CNN으로 학습
  - 매우 느림(Time-consuming)
- SPPNet은 Convolution 마지막 층에서 나온 Feature map을 분할하여 평균을 내고 고정된 크기로 만든다

### Spatial Pyramid Pooling
- BoW(Bag of Word)라는 개념을 사용한 것인데, 간단하게 말하자면 특정 개체의 분류에 '굵은 소수'의 특징이 아닌 '작은 다수'의 특징에 의존
- Spatial pyramid Matching 이라는 개념 사용<br>
![image](https://user-images.githubusercontent.com/72767245/102811340-8c58f780-4408-11eb-913f-762e62ff51b2.png)
<img src="https://user-images.githubusercontent.com/72767245/102811480-bad6d280-4408-11eb-98f9-d90e8f918582.png" width="15%"> <img src="https://user-images.githubusercontent.com/72767245/102811717-2620a480-4409-11eb-8041-7ade0235a93a.png" width="30%">

![image](https://user-images.githubusercontent.com/72767245/102808997-95e06080-4404-11eb-97e2-34eb58aa3314.png)<br>
- 위의 진은 4x4, 2x2, 1x1 의 세가지 영역으로 제공
- 각각을 하나의 **피라미드**라고함
- 피라미드의 한칸을 **bin**이라고 함 <br>
ex) feature map: 64x64x256 -> 4x4 피라미드의 bin 크기는? 64/4 = 16, 16x16 
###### 마지막 Pooling Layer를 SPP(Spatial Pyramid Pooling)로 대체 + 내부적으로 Global Max Pooling 사용
###### 실제 실험에서 저자들은 1x1, 2x2, 3x3, 6x6의 총 4개의 피라미드로 SPP 적용
###### -> 분할하는 크기만 동일하면 똑같은 크기의 Vector가 출력됨

**수학적으로 계산** <br>
feature map: 64x64x256 <br>
채널의 크기: k = 256 <br>
bin의 갯수: M <br>
(16+4+1) = 21 = M <br>
- output의 차원 kM차원 벡터
<br>

##### SPP-layer는 Conv layer에서 추출된 Feature를 입력으로 받고 이를 Spatial bin이라고 불리는 1*1, 2*2, 3*3, 4*4 등의 filter들로 잘라내어 pooling
###### Convolution 마지막 층에서 나온 Feature Map을 분할하여 평균을 내고 고정 크기로 만듦
###### SPP layer는 쉽게 말해서 이미지의 사이즈와 상관없이 특징을 잘 반영할 수 있도록 여러 크기의 bin을 만들고 그 bin값을 활용하는 구조입니다. 

### 차이점 도식화
![image](https://user-images.githubusercontent.com/72767245/102813779-cb894780-440c-11eb-8ed9-e01f518a7ab0.png)

### SPPNet의 장점
- 속도 향상
- 고정된 이미지만을 필요로 하지 않는다.

### SPPNet의 한계점
- R-CNN과 같은 학습 파이프라인을 가지고 있기 때문에 multi-stage로 학습이 진행 (fine-tuning, SVM training, Bounding Box Regression) // end-to-end 방식이 아님
  - 저장공간을 요구
  - 학습이 여전히 느림
- 여전히 최종 classification은 binary SVM, Region Proposal은 Selective Search 사용
  - SPP Pooling 이후에도 2천개의 RoI에 대해서 classification 연산을 적용하는 부분은 동일하게 적용
- CNN의 파라미터가 학습이 되지 않기 때문에 Task에 맞는 fine-tuning이 어려워짐 // pretrained model

<br>
R-CNN과 SPPnet의 장점을 가져오고 단점을 보완하고자 제안된 결과물이 바로 **Fast R-CNN**

# Fast R-CNN


참고
<url>https://woosikyang.github.io/fast-rcnn.html</url>
<url>https://yeomko.tistory.com/13</url>
<url>https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html</url>
