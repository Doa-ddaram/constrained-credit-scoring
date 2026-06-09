**Project Overview**
- **Purpose**: 이 폴더는 신용점수(loan approval) 모델들에 대해 MILP(정수선형계획)를 이용한 검증과 분석을 제공합니다. 주요 스크립트 `verification_milp.py`는 XGBoost와 간단한 MLP 모델을 MILP로 인코딩하여 다음과 같은 속성을 자동으로 검증합니다: monotonicity, individual fairness, local robustness, recourse, 그리고 compare.

**Quick Start**
- **Dependencies**: 프로젝트 루트의 [requirements.txt](requirements.txt)을 확인해 필요한 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

- **예제 실행**: 기본적으로 `verification_milp.py`는 저장된 모델 파일을 `saved_weights/`에서 읽습니다. 간단한 실행 예:

```bash
python verification_milp.py --property compare
python verification_milp.py --property local_robustness --sample-index 0 --epsilon 0.01
python verification_milp.py --property recourse --sample-index 0
```

**주요 파일**
- **verification_milp.py**: MILP 기반 검증 도구의 본체. 모델을 MILP로 인코딩하고 PuLP를 통해 최적화 문제를 풀어 위배 사례(violations)나 최소 변경(recourse)을 찾습니다. 상세 동작은 아래 섹션을 참고하세요. (파일: [verification_milp.py](verification_milp.py))
- **dataset.py**: 데이터 로드 및 전처리 유틸리티로, `verification_milp.py` 실행 시 특성 이름(`feature_names`)과 테스트 입력을 제공합니다. (파일: [dataset.py](dataset.py))
- **train_baseline_mlp.py**, **train_xgboost.py**: 모델 학습 스크립트로, 결과 모델은 `saved_weights/`에 저장됩니다. (파일들: [train_baseline_mlp.py](train_baseline_mlp.py), [train_xgboost.py](train_xgboost.py))
- **saved_weights/**: 학습된 모델 파일들(`best_baseline_mlp.pth`, `best_xgboost.json`)이 위치하는 기본 폴더.

**핵심 개념 및 구현 요약**
- **입력 쌍(x_a, x_b)**: MILP 문제는 두 입력 벡터 `x_a`(변경된/대안 입력)와 `x_b`(참조/원본)를 변수로 생성합니다. 범주형(one-hot) 피처는 바이너리 변수로 처리되고 그룹 합계가 1이 되도록 제한합니다.
- **모델 인코딩**:
  - `xgboost` : 각 Decision Tree를 리프 선택(binary) 변수와 분기 제약으로 표현하여 트리 예측을 선형식으로 합산합니다.
  - `mlp` : 각각의 선형층을 그대로 변수와 선형 제약으로 인코딩하고, ReLU는 Big-M 방식(활성화 바이너리 + 연속 변수)을 사용해 MILP로 모델을 근사합니다.
- **속성 제약(Property constraints)**:
  - `monotonicity`: 특정 피처가 증가(혹은 감소)할 때 출력이 증가해야 한다는 제약을 추가하여 그 제약을 위배하는 `x_a`를 찾습니다.
  - `individual_fairness`: 민감값(sensitive features)이 동일할 때 다른(비민감) 피처가 같음을 강제하여, 출력 차이를 찾습니다.
  - `local_robustness`: 주어진 원본 `original_x`로부터 L∞-ball(반경 `epsilon`) 내에서 출력의 부호(승인/거절)가 뒤집히는 반례를 찾습니다.
  - `recourse`: 변화 가능성(immutable, increase-only, decrease-only, flexible)을 기반으로, 가중 L1 거리(특성별 비용으로 가중)를 최소화하여 승인 결과를 얻을 수 있는 최소 변경안을 찾습니다.
- **목적 함수 및 해석**: `recourse`는 가중 L1 거리 최소화(행동비용 최소화)를 목적함수로 설정하고, 다른 속성은 위배 사례(설정한 부등식)를 만족시키는 해를 찾는 것을 목표로 합니다. 해가 `optimal` 또는 `feasible`이면 counterexample 혹은 recourse plan을 출력합니다.

**설정 가능한 주요 인자(간단 요약)**
- `--property`: 검증할 속성 (monotonicity, individual_fairness, local_robustness, recourse, compare)
- `--xgboost-path`, `--mlp-path`: 모델 경로 (기본은 `saved_weights/` 내부 파일)
- `--target-feature`: monotonicity 검사 대상 피처
- `--sensitive-features`: individual fairness 대상 피처 목록
- `--epsilon`: local robustness의 L∞ 반경
- `--bound-limit`: 입력 변수의 상/하한 (무분별한 값 제한)
- `--timeout-seconds`: MILP solver 시간 제한

**실행 결과 해석**
- 출력에 포함되는 핵심 항목들: `status`(Optimal/Feasible/ infeasible 등), `solve_seconds`, `wall_seconds`, 그리고 위배가 발견되었을 때의 `xA`, `xB`, `outputA`, `outputB`.
- `recourse`가 `Optimal`이면 추천 변경사항(각 피처별 현재값/권고값/변화량)과 함께 변경 불가능한(immutable) 피처를 명시합니다.

**제약 및 참고사항**
- MILP 기반 검증은 이론상 NP-hard이며, ReLU 활성화에 대한 Big-M 인코딩과 tree-encoding으로 인해 이진변수 수가 급증할 수 있습니다. 큰 모델이나 많은 피처에서는 풀이 시간이 크게 증가할 수 있습니다.
- `requirements.txt`의 solver(예: HiGHS, CBC 등)가 설치되어 있으면 성능과 시간 제한을 개선할 수 있습니다.

**다음 단계 제안**
- 더 큰 모델을 다루려면 Relaxation 기법(SDP/LP-relaxation) 또는 구간 바운딩(IBP) 등의 근사 기법 도입을 고려하세요.
- 출력 포맷(예: JSON 저장) 추가로 자동화된 리포트 파이프라인을 만들 수 있습니다.

---
작성자: 자동 생성 문서 (요약본). 질문이나 추가 보강을 원하시면 알려주세요.
# Prepare models (optional)
If you don't yet have trained models in `saved_weights/`, run the helper script which will invoke the training scripts provided in this folder. This is optional but convenient for getting the required model files before running verification.

```bash
python prepare_models.py
```

This script will run `train_baseline_mlp.py` and `train_xgboost.py` if the expected files (`saved_weights/best_baseline_mlp.pth`, `saved_weights/best_xgboost.json`) are missing.

# constrained-credit-scoring