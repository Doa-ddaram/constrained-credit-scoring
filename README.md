# 🏦 Constrained Credit Scoring: MILP 기반 신용평가 모델 검증 프레임워크

## 📖 프로젝트 개요 (Overview)
본 프로젝트는 머신러닝 기반의 **신용 평가 및 대출 심사(Loan Approval) 모델**이 신뢰할 수 있고 공정하게 작동하는지 검증하는 프레임워크입니다. 

트리 기반 모델(XGBoost)과 신경망 모델(MLP)을 혼합 정수 선형 계획법(MILP, Mixed-Integer Linear Programming)으로 인코딩하여, 모델이 사전에 정의된 정형적 속성(단조성, 공정성 등)을 완벽히 준수하는지 수학적으로 증명하거나 그 위배 사례(Counterexample)를 자동으로 탐색합니다.

---

## ✨ 핵심 검증 속성 (Key Properties)

`verification_milp.py`를 통해 다음 5가지 주요 속성을 검증할 수 있습니다.

1. **단조성 (Monotonicity)**
   - 자산이나 소득 등 긍정적인 요인이 증가했을 때, 대출 승인 확률이 불합리하게 감소하는 역전 현상이 없는지 검증합니다.
2. **개별 공정성 (Individual Fairness)**
   - 나이, 성별 등 '민감한 피처(Sensitive Features)'만 다르고 나머지 조건이 완벽히 동일한 두 사용자에 대해 모델이 다른 판정을 내리는 차별 사례를 찾습니다.
3. **지역적 강건성 (Local Robustness)**
   - 입력 데이터에 미세한 노이즈($L_\infty$-ball 반경 내)가 추가되었을 때, 승인/거절 결과가 갑작스럽게 뒤집히는 적대적 반례를 탐색합니다.
4. **역조치 제안 (Counterfactual Recourse)**
   - 대출이 '거절'된 고객을 대상으로, '승인'을 받기 위해 현실적으로 변경해야 하는 최소한의 조건(예: 신용카드 사용액 감소 등)을 행동 비용 기반으로 산출합니다.
5. **모델 비교 (Model Comparison)**
   - 구조가 다른 두 모델(XGBoost와 MLP)이 동일한 입력에 대해 서로 다른 결과를 내는 경계 영역을 찾아내어 모델 전환 시의 리스크를 분석합니다.

---

## 📂 파일 구조 (Project Structure)

```text
constrained-credit-scoring/
├── saved_weights/           # 학습이 완료된 모델 가중치 파일 저장소
│   ├── best_baseline_mlp.pth
│   └── best_xgboost.json
├── dataset.py               # 데이터 로드, 전처리 및 테스트 샘플 제공
├── prepare_models.py        # 모델 작동 스크립트
├── train_baseline_mlp.py    # MLP 모델 학습 스크립트
├── train_xgboost.py         # XGBoost 모델 학습 스크립트
├── verification_milp.py     # MILP 검증 엔진 (메인 스크립트)
└── requirements.txt         # 파이썬 패키지 명세
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 의존성 설치
최적화 문제 풀이를 위한 PuLP 및 필요한 머신러닝 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. (선택) 모델 사전 학습
`saved_weights/` 폴더에 모델 파일이 없는 경우, 아래 명령어로 모델을 먼저 학습시킵니다.
```bash
python prepare_models.py
```

### 3. 검증 스크립트 실행
검증하고자 하는 속성(`--property`)을 지정하여 메인 엔진을 실행합니다.

```bash
# 1. 두 모델(XGBoost vs MLP) 간의 예측 불일치 사례 탐색
python verification_milp.py --property compare

# 2. 특정 샘플(index 0)에 대해 노이즈(epsilon 0.01) 내 결과가 뒤집히는지 강건성 검사
python verification_milp.py --property local_robustness --sample-index 0 --epsilon 0.01

# 3. 특정 샘플(index 0)에 대한 최소 변경 승인 조건(Recourse) 탐색
python verification_milp.py --property recourse --sample-index 0
```

---

## ⚙️ 주요 실행 옵션 (Arguments)

| 옵션명 | 설명 | 예시 |
| :--- | :--- | :--- |
| `--property` | 검증할 속성을 선택합니다. | `compare`, `recourse`, `monotonicity` 등 |
| `--sample-index` | 로컬 검증 시 기준이 될 테스트 데이터의 인덱스입니다. | `0`, `15` |
| `--target-feature` | 단조성 검사 시 대상이 되는 피처 이름입니다. | `credit_amount` |
| `--sensitive-features` | 공정성 검사 시 보호해야 할 민감 피처 목록입니다. | `age,gender` |
| `--epsilon` | 강건성 검사 시 허용할 변화 반경($L_\infty$)입니다. | `0.01` |
| `--timeout-seconds` | 탐색을 수행할 최대 제한 시간(초)입니다. | `300` |

---

## 🛠 작동 원리 (How it Works)

본 프레임워크는 블랙박스 모델을 MILP 제약식으로 투명하게 변환합니다.
- **XGBoost**: 각 의사결정 나무의 노드 분기 조건을 이진 변수(Binary Variable)로 매핑하고 리프 노드의 가중치를 선형 결합합니다.
- **MLP**: 선형 계층은 일차 방정식으로, 비선형 활성화 함수(ReLU)는 이진 변수를 활용한 **Big-M 수식**으로 근사하여 완벽한 수학적 모델을 구성합니다.

이후, 솔버(Solver)가 지정된 속성의 위배 조건(예: $y_{original} \neq y_{perturbed}$)을 목적 함수로 삼아 해를 탐색합니다. 해가 존재하면(`Optimal`/`Feasible`) 위배 사례가 있는 것이며, 해를 찾을 수 없으면(`Infeasible`) 해당 속성에 대해 모델이 안전함을 수학적으로 증명한 것입니다.