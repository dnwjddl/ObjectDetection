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
![image](https://user-images.githubusercontent.com/72767245/103206758-e9403a80-493f-11eb-8f28-cb47ca386b92.png)
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
