# Faster R-CNN
Selective Search를 사용하여 계산해왔던 Region Proposal 단계를 Neural Network 안으로:  ```end-to-end object Detection```  
모든 단계 합쳐서 5fps라는 빠른 속도. Pascal VOC(Dataset) 기준으로 78.8%의 성능

 <br>
 두가지의 모듈로 구성됨
 1. region proposal을 하는 deep conv network
 2. 제안된 영역을 사용하는 Fast R-CNN

## Faster R-CNN 흐름

Faster R-CNN의 가장 핵심적인 구조 : ```RPN``` <br>
Fast R-CNN의 구조를 그대로 계승하면서 selective search를 제거하고 RPN을 통해서 RoI를 계산 > GPU를 통한 RoI의 계산이 가능해졌으며, RoI계산 역시도 학습시켜 정확도를 높일 수 있음 <br>
RPN은 Selective Search가 2000개의 RoI를 계산하는 데 반해 800개 정도의 RoI를 계산하면서도 더 높은 정확도 <br>

![image](https://user-images.githubusercontent.com/72767245/103479929-5d814f00-4e14-11eb-86ac-23783ec39294.png)

- Feature Map를 먼저 추출
- RPN에 전달
- RoI를 계산
- 여기서 얻은 RoI로 RoI Pooling 을 진행
- Classification을 진행하여 Object Detection을 수행

## RPN (Region Proposal Network)

### RPN 동작 방법
1. CNN을 통해 뽑아낸 feature Map을 입력으로 받는다. (H x W x C)
2. feature map에 3x3 convolution을 256 혹은 512 채널만큼 수행 (위 그림에서는 intermediate layer에 해당)<br>
 이때 padding을 1로 설정하여 H x W가 보존될 수 있도록 해줌. intermediate layer 수행 결과 H x W x 256 or H x W x 512 크기의 두번째 feature Map을 얻음
 -------------------------------------------
3. 두번째 feature map을 입력받아서 classification과 bounding box regression 예측 값을 계산 <br>
  - Fully Connected Layer가 아니라 1x1 컨볼루션을 이용하여 계산하는 Fully Convolution Network의 특징을 갖음
  - 입력 이미지의 크기에 상관없이 동작할 수 있도록 하기 위함

4. ```Classification```
  - RPN은 Conv로부터 얻은 feature map의 어떠한 사이즈의 이미지를 입력하고 출력으로 직사각형의 object score와 object proposal을 뽑아냄 
