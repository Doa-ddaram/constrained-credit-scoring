import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. 데이터 로드 (OpenML ID: 31 = German Credit)
# 1. Load data (OpenML ID: 31 = German Credit)
# by https://www.openml.org/search?type=data&exact_name=credit-g&sort=runs&status=active&id=31
print("Loading data...")
data = fetch_openml(data_id=31, as_frame=True, parser='auto')
X = data.data
y = data.target

# 2. 타겟 변수 전처리 ('good' = 1 (대출 승인), 'bad' = 0 (대출 거절))
# 2. Preprocess target labels ('good' = 1 (loan approved), 'bad' = 0 (loan rejected))
y = y.map({'good': 1, 'bad': 0}).astype(int)

# 3. 특성 분리 (수치형 vs 범주형)
# 3. Split features into numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['category', 'object']).columns

# 4. 전처리 파이프라인 구성
# 4. Build the preprocessing pipeline
# SMT 솔버(Marabou) 검증 시 수식 범위를 제한하기 위해 StandardScaler 적용이 유리합니다.
# Applying StandardScaler is helpful to keep the expression range limited during SMT solver (Marabou) verification.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# 데이터 변환 적용
# Apply data transformation
X_processed = preprocessor.fit_transform(X)

# 5. PyTorch 텐서(Tensor)로 변환
# 5. Convert to PyTorch tensors
X_tensor = torch.tensor(X_processed, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) # [batch_size, 1] 형태로 맞춤
# Match the shape to [batch_size, 1]

# 6. 학습/테스트 데이터셋 분리 (8:2 비율)
# 6. Split into train/test datasets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# 7. PyTorch DataLoader 생성
# 7. Create PyTorch DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 미니 배치 크기는 32로 설정 (필요에 따라 조정 가능)
# Set the mini-batch size to 32 (can be adjusted as needed)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"data preprocessing completed! Final feature (input node) count: {X_processed.shape[1]}")