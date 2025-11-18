# AI-X-Deeplearning-project

### Title: 태아 검사 데이터를 이용한 건강 상태 분류

### Members: 
          김서현, 전기공학전공, doodne2001@hanyang.ac.kr
          최준영, 전기공학전공, jyoki@hanyang.ac.kr
          최태욱, 전기공학전공, terry03@hanyang.ac.kr


## Index
####           I. Proposal
####           II. Datasets
####           III. Methodology
####           IV. Code guide
####           V. Results & Analysis
####           VI. Conclusion

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## I. Proposal (Option A)

### Motivation:

최근 AI와 딥러닝 기술의 발전은 다양한 분야에 큰 변화를 가져오고 있습니다. 그중에서도 의료 분야에서는 바이오 신호와 생체 데이터 분석을 통해 진단과 예측의 정확도가 크게 향상되고 있습니다. 특히, 태아의 건강 상태를 정량적으로 평가하고 분류하는 문제는 산모와 태아의 안전을 위해 매우 중요합니다.

과거에는 의료진이 여러 신호와 임상 데이터에 기반하여 수동적인 판단을 내렸으나, 인공지능 및 딥러닝 모델을 활용하면 방대한 데이터를 효율적으로 분석하고 복잡한 변수 간의 관계를 효과적으로 파악할 수 있습니다. 이를 통해 의료진의 부담은 줄이고, 조기 진단 및 위험 감지 가능성을 높일 수 있습니다.

### Our Goal:

이번 프로젝트에서는 "Fetal Health Classification" 데이터셋(21개의 바이오 신호 특성, 1개의 결과 변수)을 활용하여, 딥러닝 기반으로 태아 건강 상태(정상, 의심, 병리적)를 분류하는 모델을 설계 및 실험해보고자 합니다. 이 프로젝트는 딥러닝 기법을 실제 건강 데이터에 적용해보고, 의료 데이터 특유의 클래스 불균형과 특성 해석 등 현장에서의 복잡한 문제도 함께 경험하는 데 목적이 있습니다. 이렇게 얻은 경험은 앞으로 다양한 의료 데이터 분석 및 지능형 진단 시스템 개발에 도움이 될 것으로 기대됩니다.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## II. Datasets

본 프로젝트에서 사용할 Dataset은 다음과 같습니다. (출처: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

| 피처명                    | 설명                                 | 예시                                            |
|---------------------------|--------------------------------------|-------------------------------------------------|
| Baseline value                  | 직원 ID                              | 1001, 1002, ...                                 |
| Accelerations                   | 하루 평균 근무 시간                   | 8, 9, 10                                        |
| Fetal_Movement                  | 근무 장소                            | Home(재택), Office(사무실), Hybrid(혼합)         |
| Uterine_Contractions            | 업무 압박(강도)                      | High, Medium, Low                               |
| Light_Decelerations             | 관리자 지원 수준                     | Excellent, Good, Poor                           |
| Severe_Decelerations            | 수면 습관                            | Good, Average, Poor                             |
| Prolonged_Decelerations         | 운동 습관                            | Regular, Occasionally, None                     |
| Abnormal_Short_Term_Variability | 직무 만족도                          | High, Medium, Low                               |
| Mean_Value_of_Short_Term_Variability| 워라밸(일과 삶의 균형)               | Yes(균형 유지), No(균형 미흡)                   |
| Percentage_of_Time_With_Abnormal_Long_Term_Variability| 사교성 정도                          | Yes(활발), No(비활발)                           |
| Mean_Value_of_Long_Term_Variability| 가족과 동거 여부                     | Yes(동거), No(미동거)                           |
| Histogram_Width         | 근무/거주 지역(주 거주지)           | Delhi, Pune, Hyderabad, Karnataka 등            |
| Histogram_Min                | 직원 ID                              | 1001, 1002, ...                                 |
| Histogram_Max                 | 하루 평균 근무 시간                   | 8, 9, 10                                        |
| Histogram_Number_of_Peaks               | 근무 장소                            | Home(재택), Office(사무실), Hybrid(혼합)         |
| Histogram_Number_of_Zeros      | 업무 압박(강도)                      | High, Medium, Low                               |
| Histogram_Mode           | 관리자 지원 수준                     | Excellent, Good, Poor                           |
| Histogram_Mean          | 수면 습관                            | Good, Average, Poor                             |
| Histogram_Median       | 운동 습관                            | Regular, Occasionally, None                     |
| Histogram_Variance | 직무 만족도                          | High, Medium, Low                               |
| Histogram_Tendancy       | 운동 습관                            | Regular, Occasionally, None                     |
| Fetal_Health | 직무 만족도                          | High, Medium, Low                               |
