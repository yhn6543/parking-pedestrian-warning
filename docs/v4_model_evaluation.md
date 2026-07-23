# YOLO26n v4 데이터 정제와 모델 평가

이 문서는 v3→v4 데이터 정제 과정과 세 YOLO26n 모델의 공통 test 비교, v4 모델의 미탐 분석을 함께 기록합니다. 모델 가중치와 원본 데이터셋은 저장소에 포함하지 않습니다.

## 1. v3→v4 데이터 정제

| 항목 | v3 | v4 |
|---|---:|---:|
| 전체 이미지 | 9,600 | 7,575 |
| Train | 7,680 | 5,909 |
| Validation | 960 | 763 |
| Test | 960 | 903 |
| 제거 이미지 | - | 2,025 |
| 제거 비율 | - | 21.1% |
| retained BBox 편집 이미지 | - | 336 |
| 비어 있지 않은 편집 이미지 | - | 310 |
| 명시적 빈 라벨 이미지 | - | 26 |
| 전체 BBox | 25,817 | 24,760 |

### 제외 기준

- 사람인지 객관적으로 판별하기 어려운 형상
- 해상도가 지나치게 낮아 단일 사람 BBox 경계를 일관되게 지정하기 어려운 객체
- 라벨 오류가 심하고 신뢰할 수 있게 수정하기 어려운 이미지
- 프로젝트 목표에서 크게 벗어난 극단적인 비주차·군중 장면
- 심하게 유사하거나 중복된 이미지

### 유지 기준

- 작더라도 사람임이 명확한 객체
- 일부가 가려졌더라도 사람과 경계를 합리적으로 판별할 수 있는 객체
- 차량 사이, 주차장, 도로처럼 목표 환경에 가까운 장면
- 사람이 없는 정상적인 주차장·도로 장면
- 사람이 없는 장면의 명시적 0바이트 라벨

작거나 검출하기 어렵다는 이유만으로 샘플을 제거하지 않았습니다. 모델 예측은 사람 존재 여부와 라벨 품질을 확인하는 보조 자료로만 사용했고, 예측 성공·실패나 confidence를 `KEEP`/`DROP` 기준으로 사용하지 않았습니다.

### 분할 독립성

v4는 v3의 train/validation/test 배정을 그대로 보존한 뒤 `DROP` 이미지만 제거했습니다. 새 무작위 분할, split 간 이동, test→train 이동은 없었습니다. 공통 test 평가와 미탐 분석 이후에는 test 이미지를 추가로 제거하지 않았고, 미탐이 발생한 404장도 test에 그대로 남아 있습니다.

따라서 이 작업은 test 결과를 이용해 test를 쉽게 만드는 절차가 아니라, 고정된 분할 안에서 수동 데이터 품질 기준을 적용한 정제 과정입니다.

## 2. 공통 test 903장 모델 비교

세 모델을 동일한 v4 test 903장과 동일한 평가 옵션으로 다시 평가했습니다.

| 조건 | 값 |
|---|---|
| Task / class | Object detection / `person` |
| Test images | 903 |
| GT instances | 3,895 |
| Image size | 768 |
| Batch | 16 |
| Device | CUDA GPU |
| Confidence | 0.001 |
| NMS IoU | 0.7 |
| Max detections | 300 |
| Augmentation | False |
| Plots / JSON export | False |

평가 과정에서 학습이나 resume은 실행하지 않았습니다.

| Model | Training data | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLO26n Pretrained | COCO pretrained | 0.7446 | 0.4606 | 0.5364 | 0.2781 |
| YOLO26n v3 | v3, 9,600 images | 0.8452 | 0.6650 | 0.7638 | 0.4999 |
| YOLO26n v4 | v4, 7,575 images | 0.8482 | 0.6223 | 0.7107 | 0.4650 |

v4−v3 변화량:

| Metric | 변화량 |
|---|---:|
| Precision | +0.2951%p |
| Recall | -4.2619%p |
| mAP50 | -5.3170%p |
| mAP50-95 | -3.4863%p |

![동일 test 903장 모델 비교](assets/common_test_903_model_comparison.png)

v4는 pretrained 모델보다 모든 지표가 높고 v3보다 Precision이 소폭 높았지만, Recall과 두 mAP는 v3보다 낮았습니다. 정제된 데이터의 라벨 일관성과 목적 적합성은 개선됐지만, 대량으로 제거된 밀집·소형 사람 샘플이 공통 test의 분포와 맞물려 Recall 저하에 영향을 주었을 가능성이 있습니다. 이는 관측 결과에 대한 해석이며 인과관계를 확정하는 실험은 아닙니다.

공개 수치는 [CSV](results/common_test_903_model_comparison.csv)와 [JSON](results/common_test_903_model_comparison.json)으로도 제공합니다. 이 표는 README의 기존 validation 표와 평가 데이터가 다르므로 두 수치를 같은 결과처럼 직접 비교하면 안 됩니다.

## 3. v4 학습 진행

v4 모델은 100 epochs 동안 학습했습니다. 아래 대시보드는 학습 중 validation 지표이며 공통 test 결과가 아닙니다.

- Recall 최고 0.8214, epoch 95
- mAP50 최고 0.8917, epoch 87
- mAP50-95 최고 0.6176, epoch 88
- 마지막 epoch까지 급격한 성능 붕괴는 관찰되지 않음

![v4 학습 지표 대시보드](assets/v4_training_metrics_dashboard.png)

## 4. 고정 threshold 미탐 분석

미탐 분석은 v4 test 903장을 다음 조건으로 별도 추론해 수행했습니다.

| 조건 | 값 |
|---|---:|
| Confidence | 0.25 |
| Match IoU | 0.5 |
| NMS IoU | 0.7 |
| Image size | 768 |

| 항목 | 결과 |
|---|---:|
| Test images | 903 |
| GT | 3,895 |
| TP | 2,412 |
| FN | 1,483 |
| FP | 427 |
| Fixed-threshold Recall | 0.6193 |
| 미탐 발생 이미지 | 404 |
| 완전 미탐 이미지 | 69 |
| 부분 미탐 이미지 | 335 |
| 미탐 없는 이미지 | 499 |

이 Recall은 `confidence=0.25`의 단일 운영점에서 계산한 분석값입니다. 학습 중 validation Recall이나 `confidence=0.001`로 수행한 Ultralytics 공통 test 평가 Recall과 계산 조건이 다릅니다.

### 원본 이미지 기준 BBox 높이별 Recall

| GT BBox 높이 | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| < 16px | 116 | 0 | 116 | 0.0000 |
| 16–31px | 445 | 20 | 425 | 0.0449 |
| 32–63px | 905 | 316 | 589 | 0.3492 |
| 64–127px | 920 | 697 | 223 | 0.7576 |
| ≥ 128px | 1,509 | 1,379 | 130 | 0.9139 |

![원본 이미지 기준 BBox 높이별 Recall](assets/v4_recall_by_bbox_height.png)

32px 미만의 극소형·원거리 보행자 검출은 매우 어려웠고, 64px 이상부터 Recall이 크게 개선됐습니다. 전체 FN은 작은 객체뿐 아니라 가림, 밀집, 저대비, 모션 블러가 함께 나타나는 장면에도 집중됐습니다.

## 5. 대표 미탐 사례

![대표 미탐 사례 모음](assets/v4_representative_misses.png)

| 유형 | 이미지 | GT | TP | FN | FP | 주요 요인 | 출처 / 라이선스 |
|---|---|---:|---:|---:|---:|---|---|
| 극소형·원거리 | [보기](assets/miss_examples/miss_small_person.png) | 8 | 2 | 6 | 0 | 원거리의 작은 사람 군집 | cctv-naxyo / CC BY 4.0 |
| 가림·복잡 배경 | [보기](assets/miss_examples/miss_occlusion.png) | 9 | 1 | 8 | 0 | 구조물과 작업 물품에 의한 가림 | cctv-naxyo / CC BY 4.0 |
| 군중·부분 미탐 | [보기](assets/miss_examples/miss_crowd.png) | 9 | 5 | 4 | 1 | 겹침과 원근 크기 차이 | PersonNormal / Public Domain |
| 저대비·블러 | [보기](assets/miss_examples/miss_low_contrast.png) | 5 | 1 | 4 | 1 | 낮은 선명도와 배경 혼합 | cctv-naxyo / CC BY 4.0 |
| 차량 사이 핵심 장면 | [보기](assets/miss_examples/miss_core_scenario.png) | 4 | 0 | 4 | 1 | 차량 뒤쪽의 작은 보행자 | cctv-naxyo / CC BY 4.0 |

대표 예시는 FN 수가 큰 장면만 고른 것이 아니라 프로젝트의 핵심 환경과 서로 다른 실패 유형을 보여 주도록 선정했습니다.

### 출처, 라이선스와 개인정보 확인

- cctv-naxyo 예시 4장: [cctv Dataset by dataset, Roboflow Universe](https://universe.roboflow.com/dataset-uutxr/cctv-naxyo), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 저장소의 이미지는 분석 주석을 추가한 사본입니다.
- PersonNormal 예시 1장: [PersonNormal by hatice-nursoy, Roboflow Universe](https://universe.roboflow.com/hatice-nursoy/personnormal-u5gvy), 프로젝트 페이지 표기 `Public Domain`.
- CityPersons/Cityscapes와 프로젝트 수준 라이선스 또는 이미지 원권리가 불명확한 출처는 대표 이미지 공개 대상에서 제외했습니다.
- 5장을 원본 크기로 확인한 결과 식별 가능한 얼굴이나 읽을 수 있는 차량 번호판이 없어 별도 블러 처리를 하지 않았습니다.

라이선스 표기는 각 공개 프로젝트의 메타데이터를 따릅니다. 공개 프로젝트의 라이선스 표기가 개별 이미지 원권리까지 대신 보증하는 것은 아니므로 원본 데이터나 전체 test 이미지는 배포하지 않습니다.

## 6. 제한사항과 다음 단계

- 32px 미만의 극소형 보행자와 밀집·가림 장면에서 Recall이 매우 낮습니다.
- 현재 분석은 이미지 단위 객체 검출 성능이며, temporal filtering과 위험구역 판단의 end-to-end 성능을 대신하지 않습니다.
- 주차장 CCTV 중심 데이터이므로 모든 카메라 각도, 날씨, 조명과 현장 환경을 대표하지 않습니다.
- 공통 test는 v3 test에서 v4 정제 후 남은 부분집합이며 완전히 독립적인 외부 benchmark가 아닙니다.

다음 단계는 야간·역광·우천·가림 조건의 독립 test 구축, 소형 객체를 보존하는 고해상도 또는 타일 추론 검증, 차량 사이 보행자 중심의 외부 검증 세트 구축입니다. 현재 test 미탐 이미지를 train으로 바로 재사용하지 않고, 새 학습 데이터는 별도로 수집해 다음 데이터 버전에서 split 이력을 관리합니다.
