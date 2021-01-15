# Yolo v3

- 백 본 아키텍쳐를 Darknet-19에서 Darknet-53으로 변경
- FPN처럼 다양한 크기의 해상도의 feature map을 사용하여 Bounding Box 예측
- Class 예측 시에 Softmax 사용하지 않고 개별 클래스 별로 sigmoid를 활용한 binary classification 적용

