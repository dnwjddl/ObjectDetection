#### SPPNet의 한계점
- R-CNN과 같은 학습 파이프라인을 가지고 있기 때문에 multi-stage로 학습이 진행(end-to-end 방식이 아님)
  - Fine-Tuning
  - SVM training 
  - Bounding Box Regression
-여전히 최종 classification은 binary SVM, Region Proposal은 Selective Search(Rule based)사용
  - SPP Pooling 이후에도 2천개의 RoI에 대해 classification 연산을 적용해야 됨
- CNN의 파라미터가 학습이 되지 않기 때문에 Task에 맞는 fine-tuning 불가

# Fast R-CNN

## CNN fine-tuning, bounding bos regression, classification 을 모두 하나의 네트워크에서 학습시키는 end-to-end 기법 제시
## 결과: SPPNet보다 3배 더 빠른 학습속도, 10배 더 빠른 속도를 보임, Pascal VOC 2007 데이터 셋을 대상으로 mAP 66%

