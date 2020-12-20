# R-CNN

## RCNN은 CNN을 본격적으로 이용하여 Object Detection에서 높은 성능을 보임
![image](https://user-images.githubusercontent.com/72767245/102716925-46c1ff00-4322-11eb-998f-34aafdd6eb0c.png)

### R-CNN의 학습과정
1. 이미지에 대한 후보 영역(region proposal)을 생성 (약 2000여개)
2. 각 region proposal 마다 고정된 크기 wraping/crop하여 CNN 인풋으로 사용함
3. CNN을 통해 나온 feature map을 활용하여 SVM을 통한 분류, regressor를 통한 bounding box regression 진행

세 단계의 multi-stage로 구성되어 있고, selective search에 해당하는 region proposal 만큼 CNN을 돌려야 하며 큰 저장 공간을 요구하며 무엇보다도 느리다는 단점 존재

R-CNN의 단점을 보완하고자 제안된 연구

# SPPNet
