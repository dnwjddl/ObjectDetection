# Yolo v2
9000종류의 물체를 구분 할 수 있어짐  
(이전까지 Object Detection 분야에서 가장 많이 사용되었던 데이터 셋인 coco가 약 80종류의 클래스를 가짐)  

## BETTER

```Anchor Box``` -> ```Bounding Box```

1. BN 적용 (Dropout Layer 제거)
2. 높은 해상도 이미지로 백본 CNN 네트워크 fine tune(기존 yolo는 VGG모델 가지고 옴)
3. Anchor Box 개념 적용하여 학습 안정화 (기존 YOLO에서는 Fully connected Layer가 아닌 Fully Convolutional Network 형태로 prediction 계산, 앵커박스 개념 도입, 출력에서 7x7x30: 5차원벡터로 표기된 박스 2개, 20개의 클래스에 대한 score 값)  
5차원의 박스를 예측할때 (x,y,w,h,p) 사전에 박스는 어떠한 형태일 것이다 라는 사전정보 없이 그냥 박스를 prediction 하는 것  
4. Dimension Cluster 
사전에 크기와 비율이 모두 결정되어 있는 박스를 전제하고, 학습을 통해 박스의 위치나 크기를 세부조정 (처음부터 중심점의 좌표와 너비, 높이를 결정하는 것보다 훨씬 더 안정적)  
앵커박스는 적당히 직관적인 크기의 박스로 결정하고 비율을 1:1, 1:2, 2:1로 설정하는 것이 일반적이지만 yolov2에서는 여기에 learning algorithm을 적용  
coco데이터 셋의 바운딩 박스에 K-means clustering을 적용  
그 결과 앵커 박스를 5개로 설정하는 것이 precision과 recall 측면에서 좋은 결과를 냄  

5. Direct Location Prediction  
이렇게 결정한 앵커박스에 따라서 하나의 셀에서 5차원 벡터로 이루어진 바운딩 박스를 예측  
(tx, ty, tw, th, to)를 학습을 통해서 예측하게 되며, 이를 아래와 같은 방식을 적용하여 바운딩 박스를 구함  
기존 yolo가 그리드의 중심점을 예측했다면, yolov2에서는 left top 꼭지점으로부터 얼만큼 이동하는 지를 예측  

6. Fine-Grained Features  
기존의 yolo는 CNN을 통과한 마지막 레이어의 feature map만 사용하여 작은 물체에 대한 정보가 사라진다는 비판  
yolo v2에서는 상위 레이어의 피쳐맵을 하위 피쳐맵에 합쳐주는 passthrough layer 도입  

7. Multi-Scale Training  



# Yolo v3

- 백 본 아키텍쳐를 Darknet-19에서 Darknet-53으로 변경
- FPN처럼 다양한 크기의 해상도의 feature map을 사용하여 Bounding Box 예측
- Class 예측 시에 Softmax 사용하지 않고 개별 클래스 별로 sigmoid를 활용한 binary classification 적용

## Darknet-53
![image](https://user-images.githubusercontent.com/72767245/104729566-f781b980-577b-11eb-93b4-91b496457ed0.png)

3x3 컨볼루션과 1x1 컨볼루션으로 이루어진 블록을 연속해서 쌓아감  
Max Pooling 대신에 컨볼루션의 stride를 2로 취해주어 feature map의 해상도를 줄임  
Skip connection을 활용하여 residual 값을 전달하고 마지막 레이어에서 AveragePooling과 Fully Connected Layer를 통과한 뒤, softmax를 거쳐 분류

![image](https://user-images.githubusercontent.com/72767245/104729589-fd779a80-577b-11eb-9b13-9758d40abbab.png)

ResNet과 큰 차이점이 없어보이지만, FPS가 훨씬 높다는 점을 강조

### Things we tried that Didn't Work
· 앵커 박스 x, y의 offset을 너비나 높이의 비율로 예측하는 기법을 시도했으나 이는 모델 학습을 불안정하게 했다.
· 바운딩 박스의 x, y를 예측하는데 비선형 활성화 함수 말고 그냥 선형 함수를 사용해보았으나 별 효과 없었다.
· 당시 SOTA였던 RetinaNet의 Focal Loss를 적용해 보았으나 그다지 효과 없었다.
· 예측한 바운딩 박스가 True라고 판단할 IoU 기준 값을 0.3부터 0.7 사이로 설정해 보았으나 그다지 효과 없었다.
