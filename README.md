# AI-X-Deeplearning-project

### Title: 태아 검사 데이터를 이용한 건강 상태 분류

### Members: 
          김서현, 전기공학전공, doodne2001@hanyang.ac.kr
          최준영, 전기공학전공, jyoki@hanyang.ac.kr
          최태욱, 전기공학전공, terry03@hanyang.ac.kr


## Index
####           📌I. Proposal
####           📌II. Datasets
####           📌III. Methodology & Code guide
####           📌IV. Results & Analysis
####           📌V. References
####           📌VI. Conclusion

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌I. Proposal (Option A)

### Motivation:

&nbsp;&nbsp;&nbsp;최근 AI와 딥러닝 기술의 발전은 다양한 분야에 큰 변화를 가져오고 있습니다. 그중에서도 의료 분야에서는 바이오 신호와 생체 데이터 분석을 통해 진단과 예측의 정확도가 크게 향상되고 있습니다. 특히, 태아의 건강 상태를 정량적으로 평가하고 분류하는 문제는 산모와 태아의 안전을 위해 매우 중요합니다.

&nbsp;&nbsp;&nbsp;과거에는 의료진이 여러 신호와 임상 데이터에 기반하여 수동적인 판단을 내렸으나, 인공지능 및 딥러닝 모델을 활용하면 방대한 데이터를 효율적으로 분석하고 복잡한 변수 간의 관계를 효과적으로 파악할 수 있습니다. 이를 통해 의료진의 부담은 줄이고, 조기 진단 및 위험 감지 가능성을 높일 수 있습니다.

### Our Goal:

&nbsp;&nbsp;&nbsp;이번 프로젝트에서는 "Fetal Health Classification" 데이터셋(21개의 바이오 신호 특성, 1개의 결과 변수)을 활용하여, 딥러닝 기반으로 태아 건강 상태(정상, 의심, 병리적)를 분류하는 모델을 설계 및 실험해보고자 합니다. 이 프로젝트는 딥러닝 기법을 실제 건강 데이터에 적용해보고, 의료 데이터 특유의 클래스 불균형과 특성 해석 등 현장에서의 복잡한 문제도 함께 경험하는 데 목적이 있습니다. 이렇게 얻은 경험은 앞으로 다양한 의료 데이터 분석 및 지능형 진단 시스템 개발에 도움이 될 것으로 기대됩니다.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌II. Datasets

본 프로젝트에서 사용할 Dataset은 다음과 같습니다. (출처: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

| 피처명                    | 설명                                 | 범위                                          |
|---------------------------|--------------------------------------|-------------------------------------------------|
| Baseline_Value                  | 태아의 평균 심박수(bpm)          | 106 ~ 160                               |
| Accelerations                   | 초당 심박수 증가 횟수                | 0.0 ~ 0.019                                   |
| Fetal_Movement                  | 초당 태아 움직임 횟수                | 0.0 ~ 0.481       |
| Uterine_Contractions            | 초당 자궁 수축 횟수             | 0.0 ~ 0.014                           |
| Light_Decelerations             | 초당 가벼운 심박수 감소 횟수         | 0.0 ~ 0.015                           |
| Severe_Decelerations            | 초당 심한 심박수 감소 횟수           | 0.0 ~ 0.001                            |
| Prolonged_Decelerations         | 초당 장기적인 심박수 감소 횟수            | 0.0 ~ 0.005       |
| Abnormal_Short_Term_Variability | 비정상적인 단기 심박 감소 횟수            | 12.0 ~ 87.0                        |
| Mean_Value_of_Short_Term_Variability| 단기 심박 변동성의 평균값           | 0.2 ~ 7.0           |
| Percentage_of_Time_With_Abnormal_Long_Term_Variability| 전체 시간 중 비정상 장기 심박 변동성이 나타나는 비율| 0.0 ~ 91.0      |
| Mean_Value_of_Long_Term_Variability| 장기 심박 변동성의 평균값        | 0.0 ~ 50.7    |
| Histogram_Width         | 단기 변동성 하스토그램의 구간 너비  | 3.0 ~ 180.0       |
| Histogram_Min                | 단기 변동성 최솟값         | 50.0 ~ 159.0                        |
| Histogram_Max                 | 단기 변동성 최댓값                  | 122.0 ~ 238.0                               |
| Histogram_Number_of_Peaks               | 단기 변동성 히스토그램의 봉우리 개수           | 0.0 ~ 18.0        |
| Histogram_Number_of_Zeros      | 단기 변동성 히스토그램에서 값이 0인 수         | 0.0 ~ 10.0             |
| Histogram_Mode           | 히스토그램의 최빈값             | 60.0 ~ 187.0                     |
| Histogram_Mean          | 히스토그램의 평균값               | 73.0 ~ 182.0                  |
| Histogram_Median       | 히스토그램의 중앙값                    |77.0 ~ 186.0        |
| Histogram_Variance | 히스토그램 값들의 분산              | 0.0 ~ 269.0                |
| Histogram_Tendancy       | 히스토그램 값들의 전반적 경향성           | -1.0 ~ 1.0                |
| Fetal_Health | 태아의 건강상태          | 1:정상, 2:의심, 3:병리적                  |

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌III. Methodology & Code guide

## III-1. 데이터 전처리 (Data Processing)<br><br>

### 1. 결측치 및 이상치 처리

&nbsp;&nbsp;&nbsp;먼저 dataset 내에서의 결측값을 점검합니다. 만약 결측값이 존재한다면 보간, 중앙값 대체 및 제거 등의 방법 중 상황에 맞는 방식을 적용합니다.

```python
print(df.info())
print(df.describe())
```

### 2. 정규화 및 표준화

&nbsp;&nbsp;&nbsp;각 특성의 값 분포가 다르므로 학습 안정성과 효율성을 위한 스케일링이 필요합니다. 분포가 비정규가 명확한 경우 Min-Max 스케일링을 사용하여 값을 0~1 범위로 매핑하고, 평균이 0, 분산이 1이 되도록 표준화를 적용할 특성에는 StandardScaler를 사용합니다. 이 과정을 통해 딥러닝 모델의 수렴 속도를 높이고 학습 안정성을 향상시킬 수 있습니다.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. 클래스 불균형 처리

&nbsp;&nbsp;&nbsp;타깃 레이블인 Fetal_Health는 정상, 의심, 병리적 클래스 간의 분포가 불균형적일 가능성이 높습니다. 그러므로 이를 보정하기 위해 SMOTE (Synthetic Minority Over-Sampling Technique) 등의 오버샘플링 혹인 클래스 가중치를 부여하는 방식 등을 고려해 볼 수 있습니다.<br><br>
  
```python
sns.countplot(x='fetal_health', data=df)
plt.show()
```

## III-2. 특성 탐색 및 탐색 (Feature Selection & Exploration)<br><br>


### 1. 상관관계 분석

&nbsp;&nbsp;&nbsp;21개 바이오 신호 특성 간 상관 행렬 계산을 통해 높은 상관관계를 보이는 지표를 식별합니다. 이 때 상관이 매우 높은 특성 쌍은 다중공선성의 원인이 될 수 있으므로, 변수 축소나 특성 선택 가능성을 평가합니다.

### 2. 히스토그램 시각화

&nbsp;&nbsp;&nbsp;각 특성의 분포를 파악하기 위해 아래와 같이 히스토그램을 그립니다. 이를 통해 데이터의 왜도, 이상치, 집중 구간 등을 파악할 수 있습니다.

### 3. 특성 중요도 평가

&nbsp;&nbsp;&nbsp;랜덤 포레스트, XGBoost 등 트리 기반 모델을 사용해 각 특성의 예측 기여도를 계산합니다. 그 후 중요 특성을 도출하여 이후 모델 설계에 반영합니다.<br><br>

```python
rf = RandomForestClassifier()
rf.fit(X_scaled, y)
importances = rf.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importances')
plt.show()
```

## III-3. 모델 구조 설계 (Model Architecture)<br><br>


### 1. 입력층

&nbsp;&nbsp;&nbsp;21개의 정규화된 바이오 신호 특성을 벡터의 형태로 입력받습니다.

```python
input_dim = X.shape[1]
```

### 2. 은닉층

&nbsp;&nbsp;&nbsp;다중 퍼셉트론 구조를 사용하며 Batch Normalization 레이어를 배치하여 내부 공변량 변화를 완하하고 학습 안정성을 개선합니다. 또한 Dropout 레이어를 삽입하여 과적합을 줄입니다.

```python
class FetalHealthMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        ...
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
```

### 3. 출력층

&nbsp;&nbsp;&nbsp;3개의 클래스(정상, 의심, 병리적)에 대한 softmax 활성화를 통해 각 클래스에 대한 예측 확률을 제공합니다.<br><br>

```python
class FetalHealthMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FetalHealthMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## III-4. 학습 및 평가 방법 (Training & Evaluation)<br><br>


### 1. 데이터 분할

&nbsp;&nbsp;&nbsp;학습, 검증용으로 데이터를 분할(8:2 비율)하여 모델의 일반화 성능을 평가합니다. 검증 세트를 통해 모델의 일반화 성능을 실시간으로 평가하고 과적합 여부를 확인합니다.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)
```

### 2. 손실 함수 및 최적화

&nbsp;&nbsp;&nbsp;다중 함수 문제이므로 Cross-Entropy Loss를 사용합니다. 최적화에는 Adam Optimizer를 사용하여 학습 속도를 개선합니다.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 3.하이퍼파라미터 튜닝

&nbsp;&nbsp;&nbsp;은닉층의 노드 수, 학습률 등 주요 하이퍼파라미터를 실험하며, 최적의 모델 구조를 찾아내기 위해 Grid Search 혹은 Optuna를 활용합니다.

```python
hidden_dim = 128
```

### 4. 성능 평가 지표

&nbsp;&nbsp;&nbsp;정확도(전체 예측의 정밀성), F1-score(클래스 불균형을 고려한 조화평균), Confusion Matrix(클래수별 오분류 분석) 등 다양한 지표로 모델의 성능을 종합적으로 평가합니다.<br><br>

```python
print(classification_report(all_labels, all_preds))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
```

## III-5. 결과 해석 및 신뢰성 확보 (Model Interpretation & Reliability)<br><br>

### 1. 모델 해석

&nbsp;&nbsp;&nbsp;SHAP(각 특성이 예측에 얼마나 기여했는지 정밀하게 분석하는 방법), LIME(개별 예측에 대해 국소적 성명을 제공하여 실제 사례별 해석 가능성을 높이는 방법) 등 모델 해석 기법을 통해 실제로 예측에 큰 영향을 미친 특성을 분석합니다.

```python
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values, X)
```

### 2. 결과 시각화

&nbsp;&nbsp;&nbsp;주요 특성의 분포, 예측 결과(혼동행렬, ROC 곡선 등)를 시각화합니다.

```python
sns.countplot(x='fetal_health', data=df)
sns.barplot(x=importances, y=X.columns)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
shap.summary_plot(shap_values, X)
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌IV. Results & Analysis

### 1. 데이터 분포 및 클래스 불균형

&nbsp;&nbsp;&nbsp;타깃 변수(fetal_health)의 클래스 분포를 확인한 결과, 정상 표본이 약 1,600개 이상으로 대부분을 차지하며, 의심 표본은 약 300개, 병적 표본은 200개 이하로 상대적으로 적게 존재합니다. 즉, 본 데이터셋은 강한 클래스 불균형(class imbalance) 구조를 가진다고 할 수 있습니다. 이러한 불균형은 모델이 정상 표본에 유리하게 학습되는 경향을 만들 수 있으며, 실제로 의심, 병적 표본과 같은 위험군 탐지 성능을 별도로 점검할 필요가 있습니다.

<img width="950" height="664" alt="image" src="https://github.com/terry03123/AI-X-Deeplearning-project/blob/244c84fba62c3936bd826e3cce0286cd111836c3/Figure%201.png">

### 2. 중요 특성(Feature Importance) 해석

&nbsp;&nbsp;&nbsp;랜덤포레스트 기반 중요 특성 그래프에서 가장 높은 비중을 보인 특성들은 variability 계열(단기·장기 변동성 관련 지표)로 확인됩니다. 그 다음으로는 histogram_mean, histogram_mode, histogram_median 등 히스토그램 기반 심박 분포 요약 통계, 그리고 prolongued_decelerations(지속 감속), accelerations(가속), baseline_value(기저 심박) 순으로 중요도가 나타나는 것을 알 수 있습니다.
따라서 해당 모델에서 태아 건강 상태를 구분할 때 가장 핵심적인 판별 신호는 심박 변동성(variability)이며, 심박 분포의 위치·형태를 요약하는 통계치와 가속/감속 수치가 그 다음 수준의 중요도를 지닌다고 해석할 수 있고, 이는 실제 CTG(태아심박검사)에서 변동성 저하와 비정상 감속 패턴이 병적 상태와 관련된다는 임상적 지식과도 부합합니다.

<img width="950" height="664" alt="image" src="https://github.com/terry03123/AI-X-Deeplearning-project/blob/main/Figure%202.png">

### 3. SHAP Interaction 해석

&nbsp;&nbsp;&nbsp;SHAP interaction plot(top 3 특성 간 상호작용)에서는 가속(accelerations) 축에서 상호작용 값의 퍼짐이 가장 크게 나타나, 가속이 다른 변수들과 결합될 때 예측에 추가적으로 큰 영향을 주는 특성임을 나타냅니다. 반면 baseline_value 및 fetal_movement는 단독 효과보다 가속 및 변동성 특성과 함께 나타날 때 예측 기여가 커지는 상호작용형 변수로 해석됩니다. 즉, 모델은 '가속이 어떤 baseline/운동성 조건에서 나타나는가?'를 중요한 패턴으로 학습하고 있다고 볼 수 있습니다.

<img width="950" height="664" alt="image" src="https://github.com/terry03123/AI-X-Deeplearning-project/blob/main/Figure%203.png">

### 4. Confusion Matrix 기반 성능 해석

&nbsp;&nbsp;&nbsp;Confusion matrix(행=실제, 열=예측)를 통해 클래스별 성능을 계산한 결과는 아래와 같습니다. (그래프 상 index 0,1,2는 각각 fetal_health의 1,2,3 클래스에 대응)

전체 Accuracy: 약 90.1% 

클래스 1(정상): Precision ≈ 94.6%, Recall ≈ 94.9% -> 정상군은 매우 안정적으로 분류됨. 

클래스 2(의심): Precision ≈ 67.2%, Recall ≈ 76.3% -> 의심군 일부가 정상(클래스 1)으로 오분류되는 경향이 존재.

클래스 3(병적): Precision ≈ 92.3%, Recall ≈ 68.6% -> 병적군을 예측할 때 정밀도는 높으나, 실제 병적 표본을 정상/의심으로 낮춰 분류하는 경우가 상대적으로 존재.

&nbsp;&nbsp;&nbsp;종합하면, 해당 모델은 정상군은 매우 잘 탐지하지만(높은 precision/recall), 의심, 병적과 같은 위험군의 재현율(recall)이 상대적으로 낮다는 것을 알 수 있습니다. 이는 데이터 불균형으로 인해 다수 클래스에 학습이 편향된 결과로 해석되며, 의료적 관점에서 중요한 위험군 누락(false negative)을 줄이기 위한 개선이 필요합니다.

<img width="950" height="664" alt="image" src="https://github.com/terry03123/AI-X-Deeplearning-project/blob/main/Figure%204.png">

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 📌V. References

자료 출처: 

<https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification><br><br>

참고문헌: 

[AI driven interpretable deep learning based fetal health classification](https://www.sciencedirect.com/science/article/pii/S2472630324000888)

[Fetal Health Classification Based on Machine Learning](https://ieeexplore.ieee.org/document/9389902)

[Early Detection of Fetal Health Conditions Using Machine Learning for Classifying Imbalanced Cardiotocographic Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC12110323/)

[Fetal health prediction using machine learning](https://pubs.aip.org/aip/acp/article-abstract/3291/1/030016/3348470/Fetal-health-prediction-using-machine-learning?redirectedFrom=fulltext)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌VI. Conclusion

&nbsp;&nbsp;&nbsp;이번 프로젝트에서 만든 MLP 기반 분류 모델은 fetal_health 데이터의 3개 클래스를 대상으로 학습되었으며, 전체 정확도 약 90% 수준의 성능을 보였습니다. Feature importance 및 SHAP 분석 결과, 단기·장기 심박 변동성(variability) 관련 지표와 히스토그램 기반 심박 분포 통계가 예측에 가장 크게 기여하는 핵심 특성으로 확인되었습니다. Confusion matrix에서도 정상 클래스는 높은 precision과 recall로 안정적으로 분류되었고, 모델이 심박 변동성과 패턴적 분포 정보를 중심으로 태아 건강 상태를 효과적으로 판별하고 있음을 시사합니다.

&nbsp;&nbsp;&nbsp;그러나 데이터의 클래스 분포가 정상군에 크게 편중된 불균형 구조를 보였고, 그 영향이 의심, 병적 등 위험군 성능에서 드러났습니다. 특히 의심군과 병적군의 recall이 상대적으로 낮아 일부 위험 샘플이 정상으로 오분류되는 경향이 관찰되었으며, 이는 의료적 판단 맥락에서 거짓 양성 신호를 증가시킬 수 있는 잠재적 한계로 해석할 수 있습니다. 또한 학습 과정에서 검증 손실 및 과적합 여부를 체계적으로 모니터링하는 절차가 제한적이어서, 현재 성능이 최적화 수준인지에 대한 추가적인 검토가 필요할 것으로 예상됩니다.

&nbsp;&nbsp;&nbsp;이러한 결과를 종합하면, 모델은 주요 생리적 신호(variability 및 분포 통계)를 합리적으로 학습했으나 클래스 불균형으로 인해 위험군 탐지에서 보수적 성향을 보인 것으로 판단됩니다. 후속 단계에서는 클래스 가중치 적용, 오버샘플링(SMOTE 등), Focal Loss와 같은 불균형 대응 기법을 도입하여 위험군 recall을 우선적으로 개선할 필요가 있다고 생각합니다. 더 나아가 threshold 조정이나 cost-sensitive 학습을 통해 “정상으로 잘못 분류되는 위험 사례”를 최소화하는 방향의 최적화가 요구되며, 이러한 개선이 이루어진다면 실제 임상적 활용 가능성과 신뢰도를 동시에 높일 수 있을 것으로 생각합니다.


