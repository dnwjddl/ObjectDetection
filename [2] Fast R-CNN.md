#### SPPNet의 한계점
- R-CNN과 같은 학습 파이프라인을 가지고 있기 때문에 multi-stage로 학습이 진행(end-to-end 방식이 아님)
  - Fine-Tuning
  - SVM training 
  - Bounding Box Regression
- 여전히 최종 classification은 binary SVM, Region Proposal은 Selective Search(Rule based)사용
  - SPP Pooling 이후에도 2천개의 RoI에 대해 classification 연산을 적용해야 됨
- CNN의 파라미터가 학습이 되지 않기 때문에 Task에 맞는 fine-tuning 불가

# Fast R-CNN
##### CNN fine-tuning, bounding bos regression, classification 을 모두 하나의 네트워크에서 학습시키는 end-to-end 기법 제시
##### 결과: SPPNet보다 3배 더 빠른 학습속도, 10배 더 빠른 속도를 보임, Pascal VOC 2007 데이터 셋을 대상으로 mAP 66%

> R-CNN의 세가지 모듈(region proposal, classification, bounding box regression)을 각각 따로따로 수행
>> (1) region proposal 추출 -> 각 region proposal별 CNN연산 -> (2)classification, (3) bounding box regression

> Fast R-CNN에서는 region proposal을 CNN level로 통과시켜 classification, bounding box regression을 하나로 묶음
>> (1) region proposal 추출 -> 전체 image CNN 연산 -> RoI projection, RoI Pooling -> (2) classification, bounding box regression


### 전체 흐름
###### SPPNet과 다른 점: step별로 쪼개어 학습을 진행하지 않고 end-to-end 방식임

![image](https://user-images.githubusercontent.com/72767245/103206715-d88fc480-493f-11eb-8320-0e33c67a3f85.png)
1-1. 전체 이미지를 미리 학습된 CNN을 통과시켜 feature map 추출 <br>
1-2. Selective Search을 통해 RoI 찾음 <br>
2. Selective Search로 찾았던 RoI를 **feature map크기에 맞춰서 projection(RoI projection)** 시킴 <br>
3. projection 시킨 RoI에 대해 **RoI Pooling** 진행, 그 결과 고정된 크기의 feature vector 추출 <br>
4. feature vector는 fully-connected Layer를 고쳐 두개의 branch로 나뉘어진다. <br>
5-1. softmax를 통과하여 해당 RoI가 어떤 물체인지 classification (SVM은 사용하지 않음)<br>
5-2. bounding box regression을 통하여 selective search로 찾은 박스의 위치 조정<br>

![image](https://user-images.githubusercontent.com/72767245/103206745-e34a5980-493f-11eb-9e74-418da0f3b198.png)

Conv feature map 생성 > 각 RoI에 대해 feature map으로 부터 고정된 길이의 벡터 출력 > FC층을 지나 각 RoI에 대한 softmax 확률값과 class별 bounding box regression offsets 출력


### RoI pooling
![image](https://user-images.githubusercontent.com/72767245/103206758-e9403a80-493f-11eb-8f28-cb47ca386b92.png) <br>
입력 이미지를 CNN을 거쳐서 **Feature map**을 추출한다.<br>
그 후 이전에 미리 Selective Search로 만들어놨던 **RoI(=Region proposal)을 feature map에 projection**시킴 <br>
추출된 feature map을 미리 정해놓은 H x W 크기에 맞게끔 그리드를 설정 <br>
각각의 칸 별로 가장 큰 값을 추출하는 **max pooling을 실시**하면 결과값은 항상 H x W크기의 feature map이 되고, **이를 펼쳐서 feature vector을 추출**하게 됨
이러한 RoI pooling을 Spatial Pyramid Pooling에서 피라미드 레벨이 1인 경우와 동일

<br>

(1) 미리 설정한 HxW크기로 만들어주기 위해서 (h/H) * (w/H) 크기만큼 grid를 RoI위에 만든다. <br>
(2) RoI를 grid크기로 split시킨 뒤 max pooling을 적용시켜 결국 각 grid 칸마다 하나의 값을 추출한다. <br>
위 작업을 통해 feature map에 투영했던 hxw크기의 RoI는 HxW크기의 고정된 feature vector로 변환된다.<br>

###### 원래 이미지를 CNN 통과시킨 후 나온 Feature map에 이전에 생성한 RoI를 projection시키고 이 RoI를 FC layer input 크기에 맞게 고정된 크기로 변형 가능 >> 따라서, 2000번 이상의 CNN연산이 필요하지 않고 1번의 CNN 연산으로 속도를 줄일 수 있다.

###### RoI는 직사각형 모양을 띄며 (x,y,h,w)의 튜플 형태로 정의
- (x,y) : 위, 왼쪽 코너를 의미
- (h,w) : 높이와 너비

### Multi Task Loss
feature vector로 classification와 bounding box regression을 적용하여 각각의 loss을 얻어내고, 이를 back propagation하여 전체모델을 학습<br>
이때, classification loss와 bounding box regression을 적절하게 엮어주는 것이 필요 > **multi task loss**라 함.
<br>
<img align="center" src="https://user-images.githubusercontent.com/72767245/103218284-f1a76e00-495d-11eb-8ba9-4c093ce08c51.png" width="43%">

#### 입력 값들 설명
입력으로는 p는 softmax를 통해서 얻어낸 K+1(K object + 1 배경)개의 확률 값, u는 해당 RoI의 ground Truth 라벨의 값
<br><img align="center" src="https://user-images.githubusercontent.com/72767245/103218500-7f835900-495e-11eb-82ca-d25b770ab0b7.png" width="20%"><br>
bounding box regression을 적용하면 이는 K+1개의 class에 대해서 각각 x,y,w,h값을 조정하는 t^k를 리턴 <br>
(RoI가 사람일 경우 박스를 이렇게 조절해라, 고양이일 경우 이렇게 조절해라 라는 값을 return) <br>
Loss Function에서는 이 값들 가운데 ground truth 라벨에 해당하는 값만 가져오며, 이는 t^u에 해당 <br>
v는 ground truth bounding box 조절 값에 해당<br>
<img align="center" src="https://user-images.githubusercontent.com/72767245/103218528-932ebf80-495e-11eb-877f-a26ceb74d6c6.png" width="21%">

##### classification loss(p와 u를 사용하여 classification loss)
<img align="center" src="https://user-images.githubusercontent.com/72767245/103218556-a3469f00-495e-11eb-8484-1ffd32083e87.png" width="22%">
##### Bounding Box Regression loss(Bounding box regression을 통해 얻은 loss)
<img align="center" src="https://user-images.githubusercontent.com/72767245/103218570-afcaf780-495e-11eb-9949-ffc6f88b0de7.png" width="40%"><br>

##### 입력으로는 정답 라벨에 해당하는 BBR 예측 값과 ground truth 조절 값을 받음
x,y,w,h 각각에 대해서 예측 값과 라벨 값의 차이를 계산한 다음, smoothL1라는 함수를 통과시킨 합을 계산합니다. <br>
<img align="center" src="https://user-images.githubusercontent.com/72767245/103218581-b8233280-495e-11eb-99a0-ebebd996a209.png" width="40%"><br>

- 예측값과 라벨 값의 차가 1보다 작거나 크거나에 따라서 L1 distance 계산
- Object Detection 테스크에 맞추어 loss function을 custom하는 것으로 볼 수 있다.
- 라벨 값과 차이가 지나치게 차이가 많이 나는 outlier 에측값들이 발생하였고, 이들을 그대로 L2 distance로 계산하여 적용할 경우 gradient가 explode해버리는 현상이 발생 -> 이를 방지하기 위해 smoothL1 distance 추가

## 최종 손실 함수
![image](https://user-images.githubusercontent.com/72767245/103233950-bcad1280-4981-11eb-9f63-866a77ddb003.png)<br>
- Fast R-CNN은 두개의 출력층
  - 분류의 경우, 각 RoI별 클래스에 속할 사후 확률 값을 출력
  - 회귀의 경우, bounding box regression값을 출력
    - 두 출력에 대한 ground truth를 u,v로 봄

## Fine-tuning for detection
detection을 위해서 Fast R-CNN은 R-CNN&SPPNet의 region-wise sampling이 아닌 hierarchical sampling을 사용<br>
따라서 N개의 이미지를 미리 뽑고 그 중 R개의 RoI를 뽑아서 학습이 사용<br>
N = 2, R = 128응 사용  -> 약 64배의 빠른 학습이 가능<br>
수직적 구조로 인해 수렴이 늦어질수도 있지만 학습 결과 수렴 속도에 큰 영향을 미치지 않는다는 것이 언급<br>
무엇보다도 Fast R-CNN은 최종 classifier와 regression까지 단방향 단계인 single stage로 fine-tuning이 가능하다는 장점을 갖음

### Backpropagation through RoI Pooling Layer

### Initializing from pre-trained networks
각각 5개의 max pooling layer와 5~13개의 conv layer를 가진 네트워크이며 Fast R-CNN에 적용되면서 크게 3가지 변화가 생김 <br>
- 1. max pooling layer는 첫 fc layer와 호환되는 RoI pooling layer로 대체
- 2. 네트워크 마지막의 fc layer와 softmax는 앞서 언급한 바와 같이 2개의 서로 다른 layer로 대체 
- 3. 네트워크는 이미지와 region proposal 두개의 입력을 받을 수 있도록 수정



