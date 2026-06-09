import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader

GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# 원본 데이터에 맞게 컬럼명 직접 지정
# Manually specify column names to match the original dataset format.
GERMAN_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]


def load_and_preprocess_data(test_size=0.2, batch_size=32, random_state=42):
    """Prepare German Credit data and return tensors/loaders plus numpy arrays for other models."""
    # 1. 데이터 로드
    # Load the dataset from the UCI repository.
    print("Loading data from UCI repository...")

    # 공백 단위로 구분된 원본 데이터 읽기
    # Read the raw data where values are separated by spaces.
    data = pd.read_csv(GERMAN_CREDIT_URL, sep=" ", header=None, names=GERMAN_COLUMNS)
    X = data.drop("class", axis=1)
    y = data["class"]

    # 2. 타겟 변수 전처리 (UCI 원본은 1이 승인, 2가 거절을 의미함)
    # Preprocess target labels (in UCI data, 1 means approved and 2 means rejected).
    y = y.map({1: 1, 2: 0}).astype(int)

    # 3. 특성 분리 (수치형 vs 범주형)
    # Split features into numeric and categorical groups.
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # 4. 전처리 파이프라인 구성
    # Build the preprocessing pipeline.
    # SMT 솔버(Marabou) 검증 시 수식 범위를 제한하기 위해 StandardScaler 적용.
    # Apply StandardScaler to keep the expression range limited during SMT solver (Marabou) verification.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
        ]
    )

    # 데이터 변환 적용
    # Apply the fitted preprocessing pipeline to transform features.
    X_processed = preprocessor.fit_transform(X)

    # 5. PyTorch 텐서로 변환
    # Convert processed arrays to PyTorch tensors.
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    # 6. 학습/테스트 데이터셋 분리
    # Split tensors into training and test datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=random_state
    )

    # 7. PyTorch DataLoader 생성
    # Create PyTorch DataLoaders for batch iteration.
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    feature_names = preprocessor.get_feature_names_out()

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_processed": X_processed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_np": X_train.numpy(),
        "X_test_np": X_test.numpy(),
        "y_train_np": y_train.numpy().flatten(),
        "y_test_np": y_test.numpy().flatten(),
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


if __name__ == "__main__":
    bundle = load_and_preprocess_data()
    print(f"data preprocessing completed! Final feature count: {bundle['X_processed'].shape[1]}")