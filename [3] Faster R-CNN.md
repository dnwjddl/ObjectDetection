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
1. **CNN**을 통해 뽑아낸 **feature Map**을 입력으로 받는다. (H x W x C)
2. feature map에 **3x3 convolution을 256 혹은 512 채널만큼 수행**  
intermediate layer 수행 결과: H x W x 256 or H x W x 512 크기의 **두번째 feature Map**을 얻음
###### 이때 padding을 1로 설정하여 H x W 가 보존될 수 있도록 해줌  
  
3. 두번째 feature map을 입력받아서 classification과 bounding box regression 예측 값을 계산  
  ✔ Fully Connected Layer가 아니라 1x1 컨볼루션을 이용하여 계산하는 **Fully Convolution Network**의 특징을 갖음  
  ✔ 입력 이미지의 크기에 상관없이 동작할 수 있도록 하기 위함 (Fully Connected Layer은 크기가 고정되어야 함)  

4. ```Classification```  
 ✔ 1x1 컨볼루션을(2(오브젝트인지 아닌지) x 9(앵커 개수)) 채널 수 만큼 수행 > 결과 H x W x 18크기의 feature map  
 ✔ HxW상의 하나의 인덱스는 feature Map상의 좌표를 의미 그 아래 18개의 채널은 각각 해당 좌표를 앵커로 삼아 k개의 앵커 박스들이 object인지 아닌지에 대한 예측 값  
 ✔ 한번의 1x1 컨볼루션으로 HxW 개의 앵커 좌표들에 대한 예측을 모두 수행  
 ✔ 이 값들을 적절히 reshape 해준 다음 softmax를 적용하여 해당 앵커가 object일 확률 값을 얻는다  

5. ```Bounding Box Regression```  
 ✔ 1x1 컨볼루션을 (4x9) 채널 수 만큼 수행  
 ✔ Regression이기 때문에 결과로 얻은 값을 그대로 사용  

6. 앞서 얻은 값들로 RoI를 계산   
 - Classification을 통해 얻은 물체일 확률 값들을 정렬  
 - 높은 순으로 K개의 앵커만 추려냄  
 - 그 다음 K개의 앵커들에 각각 Bounding Box Regression 적용  
 - Non-Maximum-Suppression을 적용하여 RoI구하기  
 


  ✔ RPN은 Conv로부터 얻은 feature map의 어떠한 사이즈의 이미지를 입력하고 출력으로 직사각형의 object score와 object proposal을 뽑아냄 
