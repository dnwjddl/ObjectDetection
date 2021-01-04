# YOLO(You Only Look Once)

- Faster R-CNN보다 6배 빠른 속도를 가지고옴
- 1 step Object Detection 기법 제시

기존에 ```1) region proposal```, ```2)classification``` 두 단계로 나누어서 진행  
region proposal 단계를 제거하고 한번에 Object Detection을 수행하는 구조를 갖는다  

## Unified Detection
- 1. 입력 이미지를 S X S 그리드 영역으로 나눔(실제 입력 이미지 아님)
- 2. 각 그리드 영역에서 먼저 물체가 있을 만한 영역에 해당하는 B개의 Bounding Box를 예측
  - (x,y,w,h)로 나타내어 지는데 (x, y)는 bounding box의 중심점 좌표. w,h는 넓이와 높이
  - 해당 박스의 신뢰도를 나타내는 Confidence를 계산
  
  $ P_{r}(Object) * IOU_{truthpred} $  
  
$f(x) = x^2$
