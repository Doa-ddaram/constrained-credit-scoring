import torch
import torch.nn as nn
import torch.optim as optim
import os

# 1. 베이스라인 모델 정의 (ReLU만 사용)
class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super(BaselineMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_processed.shape[1]
num_trials = 10
best_accuracy = 0.0
epochs = 20

# 🌟 변경된 부분: 가중치를 저장할 디렉토리 생성
save_dir = "saved_weights"
os.makedirs(save_dir, exist_ok=True) # 폴더가 없으면 생성, 있으면 넘어감
best_model_path = os.path.join(save_dir, "best_baseline_mlp.pth")

print(f"최적의 가중치를 찾기 위해 {num_trials}번의 학습 시도를 시작합니다...\n")

for trial in range(num_trials):
    print(f"--- Trial {trial + 1}/{num_trials} ---")
    
    model = BaselineMLP(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    print(f"Training Loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            predicted = (outputs >= 0).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    if test_accuracy > best_accuracy:
        print(f">>> 🌟 최고 성능 갱신! ({best_accuracy:.2f}% -> {test_accuracy:.2f}%)")
        best_accuracy = test_accuracy
        # 생성한 폴더 경로에 가중치 저장
        torch.save(model.state_dict(), best_model_path)
        print(f">>> 가중치가 '{best_model_path}'에 저장되었습니다.")
    print("-" * 40)

print("=========================================")
print(f"모든 시도 완료. 최종 최고 정확도: {best_accuracy:.2f}%")
