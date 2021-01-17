# ObjectDetection  
논문 분석 후 직접 코드 구현  

## STUDY [이론공부]
- [[1] R-CNN_SPPNet.md](https://github.com/dnwjddl/ObjectDetection/blob/main/%EC%9D%B4%EB%A1%A0/%5B1%5D%20R_CNN_SPPNet.md)<br>
- [[2] Fast R-CNN_Faster R-CNN.md](https://github.com/dnwjddl/ObjectDetection/blob/main/%EC%9D%B4%EB%A1%A0/%5B2%5D%20Fast%20R-CNN.md)
- [[3] Faster R-CNN.md](https://github.com/dnwjddl/ObjectDetection/blob/main/%EC%9D%B4%EB%A1%A0/%5B3%5D%20Faster%20R-CNN.md)
- [[4] YOLO.md](https://github.com/dnwjddl/ObjectDetection/blob/main/%EC%9D%B4%EB%A1%A0/%5B4%5D%20YOLO.md)
- [[5] YOLOv2,v3.md](https://github.com/dnwjddl/ObjectDetection/blob/main/%EC%9D%B4%EB%A1%A0/%5B5%5D%20YOLOv2%2C3.md)

## SSD
[SSD:Single Box MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf, "paper link")

논문의 SSD Structure 직접 짜보기
### Model Structure
![image](https://user-images.githubusercontent.com/72767245/97295225-5d078d80-1892-11eb-9a2e-d59aa99d49ce.png)

### VGG19 + Auxilary Convolutional Layers
<div>
  <img src ="https://user-images.githubusercontent.com/72767245/97295251-65f85f00-1892-11eb-9657-da7936493d3e.png" width="55%">
  <img src ="https://user-images.githubusercontent.com/72767245/97295263-698be600-1892-11eb-952b-890975592614.png" width="35%">
</div>

## YOLOv3
- [yolov3_anchor](https://github.com/dnwjddl/ObjectDetection/blob/main/YOLOv3/yolov3_anchor.py)
  - bounding box 추가하여 Yolov3 model 구현
- [darknet-53](https://github.com/dnwjddl/ObjectDetection/blob/main/YOLOv3/darknet53.py)
  - model 내에 darknet-53 구현
- [utils](https://github.com/dnwjddl/ObjectDetection/blob/main/YOLOv3/utils.py)
  - 모델 구현에 필요로 하는 helper 함수들
  - IoU, NMS등 구현 (class 추출)

## cvYOLO
- cvlib을 이용한 YOLO 사용
- 총 80개의 class 식별 가능<br>

### 결과값 
<img src = "https://user-images.githubusercontent.com/72767245/97080707-5d0c5100-1638-11eb-8208-fdae6ea27dc2.jpg" width= "50%">
