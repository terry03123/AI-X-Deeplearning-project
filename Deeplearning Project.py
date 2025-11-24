# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 2. 데이터 로드 및 탐색
df = pd.read_csv('fetal_health.csv')
print(df.head())
print(df.info())
print(df.describe())
sns.countplot(x='fetal_health', data=df)
plt.show()

# 3. 데이터 전처리
X = df.drop(columns=['fetal_health'])
y = df['fetal_health']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 특성 중요도 분석 (랜덤포레스트 예시)
rf = RandomForestClassifier()
rf.fit(X_scaled, y)
importances = rf.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importances')
plt.show()

# SHAP 분석 (선택적)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values, X)

# 5. 데이터 분할 (훈련/검증)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 6. 딥러닝 모델 정의 (MLP 간단 예시)
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

# 7. 데이터셋 및 데이터로더 커스텀 클래스
class FetalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values - 1, dtype=torch.long)  # class labeling 0부터 시작

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FetalDataset(X_train, y_train)
val_dataset = FetalDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 8. 모델 학습 함수 및 평가 함수
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # 검증 절차 생략 가능 또는 추가로 구현

# 9. 모델 학습
input_dim = X.shape[1]
hidden_dim = 128
output_dim = len(np.unique(y))
model = FetalHealthMLP(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, epochs=50, optimizer=optimizer, criterion=criterion)

# 10. 검증 및 성능 평가
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 'Blues', 'YlGnBu', 'Oranges', 'plasma' 등 여러 팔레트 사용 가능
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()