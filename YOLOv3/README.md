### ```yolo_cfg```폴더
- cfg 파일과 weight을 사용하여 물체 감지
  - ```yolo.py```: 사진 객체 감지
  - ```yolo_video.py``` : 웹캠을 사용하여 실시간 객체 감지
- weight 다운로드
  - https://pjreddie.com/media/files/yolov3.weights

---
### ```train.py```
학습된 model은 checkpoint에 시간마다 저장됨

### ```Model```폴더
- YOLOv3 직접 구현한 model

### ```data``` 폴더
- VOC2012 데이터셋 transform 및 dataload

## 결과값 

![image](https://user-images.githubusercontent.com/72767245/104844285-c12a7280-5912-11eb-986e-5b49a645cc3d.png)

### 최종결과값에 대한 설명
```torch.Size([1, 10647, 85])```  
세개 다른 scale 의 output tensor 합친 값 + 80(class 갯수) + 5(위치+confidence값)  

```torch.Size([random, 8])```  

- 8개의 속성
  - 1 = batch 에서 이미지 인덱스
  - 2~5 = 꼭지점의 왼위, 오아 좌표
  - 6 = Objectness 점수
  - 7 = Maximum confidence를 가진 class 점수
  - 8 = 그 class의 index 값


![image](https://user-images.githubusercontent.com/72767245/104838191-a0efb900-58fc-11eb-9837-0612d74e3946.png)

- ```Darknet53.py``` : DarkNet-53 Layer 제작
- ```[TEST] yolov3_architecture.py``` : Layer Model 제작

#### cfg 파일 (Configuration file)

- cfg 파일은 블록단위로 네트워크의 Layout을 나타냄

저자가 제공하는 공식 cfg 파일 사용  

```python
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

저자의 cfg 파일을 토대로 ```yolov3_anchor.py``` 작성
