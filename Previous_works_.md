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


세 단계의 multi-stage로 구성되어 있음
- Selective Search을 통한 region proposal 생성
- 각 region proposal 마다 고정된 크기의 박스 추출
- 추출된 Feature map을 사용하여 SVM Classifier과 bounding box regression 진행

#### 첫번째 stage, Region Proposal
- Region proposal, 주어진 이미지에서 물체가 있을 법한 위치를 찾는 것
- Selective Search라는 룰 베이스 알고리즘을 적용하여 2천개의 물체가 있을 법한 박스를 찾는다.
- Selective Search는 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을 법한 박스를 추론
**RCNN 이후 Region Proposal 과정은 뉴럴 네트워크가 수행하도록 발전함**

#### 두번째와 세번째 stage, Feature Extraction
- Selective Search를 통해서 찾아낸 2천개의 박스 영역은 227 x 227 크기로 리사이즈(warp)
- Image Classification으로 미리 학습되어 있는 CNN 모델을 통과하여  4096크기의 특징 벡터를 추출

- 이미지넷 데이터로 미리 학습된 CNN 모델을 가지고 온 다음, fine tune 하는 방식을 취함
- Fine tune시에는 실제 Object Detection을 적용할 데이터 셋에서 ground truth에 해당하는 이미지들을 가져와 학습시켜야함

- 각 CNN 레이어 층에서 추출된 벡터로 SVM Classifier를 학습시킴

**이미지넷으로 학습된 CNN을 가지고와서, Object Detection용 데이터 셋으로 fine tuning 한 뒤, selective search 결과로 뽑힌 이미지들로부터 특징 벡터 추출**

#### 세번째 -1 stage, Classification
CNN을 통해 추출한 벡터를 각각 클래스 별로 SVM Classifier를 학습시킴
"그냥 CNN Classifier를 쓰는 것이 SVM을 썼을 때보다 mAP성능이 낮아짐. 이는 아마도 fine tuning 과정에서 물체의 위치 정보가 유실되고 무작위로 추출된 샘플을 학습하여 발생한 것으로 보임"

#### 세번째 -1 stage, Non-Maximum Suppression - IoU
SVM을 통과하여 각각의 박스들은 어떤 물체일 확률(Score)값을 가지게 됨
가장 높은 score을 가진 박스만 남기고 나머지는 제거 <Non-Maximum Suppression>

서로 다른 두 박스가 동일한 물체에 쳐져있는지 확인하는 방법(IoU: Intersection over Union)
![image](https://user-images.githubusercontent.com/72767245/102718320-dc618c80-432a-11eb-892f-996788e62473.png)
-> 두 박스의 교집합을 합집합으로 나눠준 값

**논문에서는 IoU가 0.5보다 크면 동일한 물체의 대상으로 판별**

  
#### 세번째 -2 stage, Bounding Box Regression(위치 교정)
물체가 있을 법한 위치를 찾고, 해당 물체의 종류를 판별할 수 있는 Classifier모델을 학습시킴.



**CNN을 통과하여 추출된 벡터 x,y,w,h를 조정하는 함수의 weight를 곱해서 바운딩 박스를 조정해주는 선형회귀를 학습시키는 것**

### 학습이 일어나는 부분
- 1. 이미지넷으로 이미 학습된 부분을 가져와 fine-tuning 하는 부분
- 2. SVM Classifier를 학습시키는 부분
- 3. Bounding Box Regression

### R-CNN의 단점
Selective search에 해당하는 region proposal 만큼 CNN을 돌려야함
- **큰 저장 공간을 요구**
- **속도가 느림**

R-CNN의 단점을 보완하고자 제안된 연구

# SPPNet
![image](https://user-images.githubusercontent.com/72767245/102716951-796bf780-4322-11eb-8fe3-867b3a206164.png)

## SPPNet은 R-CNN에서 가장 크게 나타나는 속도 저하의 원인인 각 region proposal 마다의 CNN feature map 생성을 보완하였고 이를 통해 속도 개선을 하게 됨
## region proposal에 바로 CNN을 적용하는 것이 아니라 이미지에 우선 CNN을 적용하여 생성한 feature map을 region proposal에 사용했기 때문

SPPnet은 Spatial Pyramid Pooling 이라는 특징을 같는 구조를 활용하여 임의 사이즈의 이미지를 모두 활용할 수 있도록 하였습니다. SPP layer는 쉽게 말해서 이미지의 사이즈와 상관없이 특징을 잘 반영할 수 있도록 여러 크기의 bin을 만들고 그 bin값을 활용하는 구조입니다. 결론적으로, SPPnet은 속도를 크게 향상 시켰고, 고정된 이미지만을 필요로 하지 않는다는 장점을 갖게 됩니다.

다만 한계점도 존재합니다. 우선 R-CNN과 같은 학습 파이프라인을 갖고 있기에 multi-stage로 학습이 진행됩니다. 따라서 저장 공간을 요구하게 되고 학습이 여전히 빠르게 진행되기는 어렵게 됩니다. 또한 위의 그림과 같이 CNN의 파라미터가 학습이 되지 못하기에 Task에 맞는 fine-tuning이 어려워집니다.

R-CNN과 SPPnet의 장점을 가져오고 단점을 보완하고자 제안된 결과물이 바로 Fast R-CNN

# Fast R-CNN


참고
<url>https://woosikyang.github.io/fast-rcnn.html</url>
