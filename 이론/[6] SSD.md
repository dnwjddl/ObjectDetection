![image](https://user-images.githubusercontent.com/72767245/108738210-b8166c00-7576-11eb-9bde-38330ef9cfc9.png)


- Yolo의 문제점은 입력 이미지를 7x7 크기의 그리드로 나누고, 각 그리드 별로 Bounding Box Prediction을 진행하기 때문에 그리드 크기보다 작은 물체를 잡아내지 못하는 문제
- 신경망을 모두 통과하면서 컨볼루션과 풀링을 거쳐 coarse 한 정보만 남은 마지막 단 피쳐맵만 사용하기 때문에 정확도가 하락함

### Multi Scale Feature Maps for Detection

![image](https://user-images.githubusercontent.com/72767245/108737658-2c9cdb00-7576-11eb-9037-25665ea53338.png)

- SSD는 300x300 크기의 이미지를 입력받아 ImageNet으로 pretrained된 VGG의 Conv5_3층까지 통과하며 Feature 추출
- 추출된 Feature map을 convolution을 거쳐 그 다음 층에 넘겨주는 동시에 Object Detection 수행
- 이전 Fully Convolution Network에서 컨볼루션을 거치면서 디테일한 정보들이 사라지는 문제점을 앞단의 feature map들을 끌어오는 방식으로 해결

---

