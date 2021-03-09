## Yolov4의 작업 요약
- 효율적이고 강력한 object detector를 1080Ti, 2080Ti와 같은 환경에서도 빠르고 정확하게 훈련 가능
- detector의 훈련과정에서 Bag-of Freebies와 Bag-of-Specials methods의 영향을 검증
- SOTA의 method를 수정하고 만듦으로써 single GPU 훈련에서도 적합하고 효율적이게 한다.

## Yolov4에서 사용한 최신 딥러닝 기법
- WRC(Weighted-Residual-Connectioins)
- CSP(Cross-Stage-Partial-Connections)
- CmBN(Cross mini-Batch Normalizations)
- SAT(Self-Adversarial-Training)
- Mish Activation
- Mosaic Data Augmentation
- Drop Block Regularization
- CIOU Loss

: MS COCO Dataset에서 AP: 43.5% (AP50: 65.7%), 65FPS(Tesla V100 그래픽 카드, 거의 실시간에 가까운 높은 FPS)달성

---
### Neural Network의 단점
- 낮은 FPS, 큰 mini-batch size로 인하여 많은 수의 GPU들이 필요하다는 단점

### Yolov4
- 일반적인 학습환경에서도 높은 정확도와 빠른 object Detector를 학습
- 1개의 GPU(GTX 1080Ti, 2080Ti)만 있으면 충분
- Detector를 학습하는 동안, 최신 **BOF, BOS기법**이 성능에 미치는 영향을 증명
- **CBN, PAN, SAM**을 포함한 기법을 활용해 single GPU Traning에 효과적

**요약: 1개의 GPU를 사용하는 일반적인 학습환경에서 BOF, BOS 기법을 적용하여 효율적이고 강력한 Object Detection을 제작**

---

## Object Detection Models
- Object Detection의 일반적인 구조(백본, Neck, Head)와 종류(1-stage, 2-stage)에 대해 소개



