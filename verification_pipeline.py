import sys, os
sys.path.insert(0, os.path.abspath("./Marabou"))
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork

import numpy as np

def verify_monotonicity(weights, biases, income_idx, input_dim, timeout_seconds=60):
    """
    단조성 제약 위반 여부를 SMT 솔버로 검증하는 함수
    
    :param weights: 파이토치 모델에서 추출한 각 층의 가중치 리스트 [W1, W2, W3]
    :param biases: 파이토치 모델에서 추출한 각 층의 편향 리스트 [b1, b2, b3]
    :param income_idx: 소득(Income) 변수가 위치한 입력 인덱스
    :param input_dim: 전체 입력 피처의 개수
    """
    print("🔍 [검증 시작] Marabou 네트워크 초기화 중...")
    network = MarabouNetwork()

    # ==========================================
    # 1. 입력 변수 선언 (두 개의 가상 입력 세트 A, B 구성)
    # 단조성을 비교하려면 '소득이 오르기 전(A)'과 '오른 후(B)' 두 가지 상태가 필요합니다.
    # ==========================================
    x_A = [network.getNewVariable() for _ in range(input_dim)]
    x_B = [network.getNewVariable() for _ in range(input_dim)]

    # ==========================================
    # 2. 탐색 범위(Bounds) 설정
    # 연산 폭발을 막기 위해, StandardScaler로 정규화된 값의 범위를 제한합니다.
    # ==========================================
    bound_limit = 3.0
    for i in range(input_dim):
        network.setLowerBound(x_A[i], -bound_limit)
        network.setUpperBound(x_A[i], bound_limit)
        network.setLowerBound(x_B[i], -bound_limit)
        network.setUpperBound(x_B[i], bound_limit)

    # ==========================================
    # 3. 가중치 적용 및 은닉층(ReLU) 구성
    # 모델 파트너(A)에게 받은 가중치를 이용해 수식 그래프를 그립니다.
    # ==========================================
    def build_forward_pass(input_vars, prefix):
        """가중치 행렬을 곱하고 ReLU를 씌워 최종 로짓 변수를 구성한다."""
        current_vars = list(input_vars)

        for layer_idx, (w_layer, b_layer) in enumerate(zip(weights, biases)):
            w_layer = np.asarray(w_layer, dtype=np.float64)
            b_layer = np.asarray(b_layer, dtype=np.float64).reshape(-1)

            if w_layer.ndim != 2:
                raise ValueError(f"Layer {layer_idx} weight must be 2D")
            if w_layer.shape[1] != len(current_vars):
                raise ValueError(
                    f"Layer {layer_idx} input dim mismatch: "
                    f"expected {len(current_vars)}, got {w_layer.shape[1]}"
                )
            if w_layer.shape[0] != b_layer.shape[0]:
                raise ValueError(
                    f"Layer {layer_idx} output dim mismatch between weight and bias"
                )

            pre_activation_vars = [network.getNewVariable() for _ in range(w_layer.shape[0])]

            # Marabou requires finite bounds; keep hidden variables in a broad but finite range.
            for var in pre_activation_vars:
                network.setLowerBound(var, -1e3)
                network.setUpperBound(var, 1e3)

            # pre_j - sum_k(W_jk * x_k) = b_j
            for j, pre_var in enumerate(pre_activation_vars):
                eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
                eq.addAddend(1.0, pre_var)
                for k, in_var in enumerate(current_vars):
                    coeff = float(w_layer[j, k])
                    if coeff != 0.0:
                        eq.addAddend(-coeff, in_var)
                eq.setScalar(float(b_layer[j]))
                network.addEquation(eq)

            is_output_layer = layer_idx == len(weights) - 1
            if is_output_layer:
                current_vars = pre_activation_vars
            else:
                post_activation_vars = [network.getNewVariable() for _ in range(w_layer.shape[0])]
                for var in post_activation_vars:
                    network.setLowerBound(var, 0.0)
                    network.setUpperBound(var, 1e3)
                for pre_var, post_var in zip(pre_activation_vars, post_activation_vars):
                    network.addRelu(pre_var, post_var)
                current_vars = post_activation_vars

        if len(current_vars) != 1:
            raise ValueError("The final layer must have exactly one logit for binary scoring")
        return current_vars[0]

    # 입력 A와 B에 대해 각각 네트워크 통과 로직 구성
    logit_A = build_forward_pass(x_A, prefix="A")
    logit_B = build_forward_pass(x_B, prefix="B")

    # ==========================================
    # 4. 검증할 속성(Property) 인코딩: "최악의 반례를 찾아라!"
    # ==========================================
    
    # 조건 4-1: 소득(income_idx)을 제외한 나머지 모든 변수는 A와 B가 완벽히 동일하다.
    for i in range(input_dim):
        if i != income_idx:
            eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(1, x_A[i])
            eq.addAddend(-1, x_B[i])
            eq.setScalar(0)
            network.addEquation(eq)

    # 조건 4-2: B의 소득이 A의 소득보다 명확히 높다. (B_income > A_income + epsilon)
    # SMT 솔버는 확실한 부등식을 좋아하므로 미세한 여유값(epsilon)을 줍니다.
    margin = 0.01 
    eq_income = MarabouUtils.Equation(MarabouCore.Equation.GE)
    eq_income.addAddend(1, x_B[income_idx])
    eq_income.addAddend(-1, x_A[income_idx])
    eq_income.setScalar(margin)
    network.addEquation(eq_income)

    # 조건 4-3 (위반 조건): 소득이 올랐음에도 불구하고, A의 대출 확률이 B보다 크거나 같다! (A_logit >= B_logit)
    eq_violation = MarabouUtils.Equation(MarabouCore.Equation.GE)
    eq_violation.addAddend(1, logit_A)
    eq_violation.addAddend(-1, logit_B)
    eq_violation.setScalar(0)
    network.addEquation(eq_violation)

    # ==========================================
    # 5. SMT 솔버 실행 (쿼리)
    # ==========================================
    print("🧠 Marabou 솔버 쿼리 진행 중... (시간이 걸릴 수 있습니다)")
    options = Marabou.createOptions(
        verbosity=0,
        timeoutInSeconds=timeout_seconds,
        numWorkers=1,
    ) # 로그 출력 최소화 + 무한 대기 방지
    exit_code, vals, stats = network.solve(options=options)

    # ==========================================
    # 6. 결과 분석 및 반환
    # ==========================================
    if exit_code == "sat" and len(vals) > 0:
        print("🚨 [SAT] 단조성 위반 반례가 발견되었습니다!")
        counter_example_A = [vals[x_A[i]] for i in range(input_dim)]
        counter_example_B = [vals[x_B[i]] for i in range(input_dim)]
        print("📌 반례 입력 A (소득 증가 전):")
        print(np.array2string(np.array(counter_example_A), precision=6, separator=", "))
        print("📌 반례 입력 B (소득 증가 후):")
        print(np.array2string(np.array(counter_example_B), precision=6, separator=", "))
        print(
            f"📈 income 변화량(B-A): "
            f"{counter_example_B[income_idx] - counter_example_A[income_idx]:.6f}"
        )
        return {"status": "SAT", "counter_example_A": counter_example_A, "counter_example_B": counter_example_B}
    if exit_code == "unsat":
        print("✅ [UNSAT] 지정된 바운드 내에서 단조성 위반 반례가 존재하지 않습니다.")
        return {"status": "UNSAT"}
    if exit_code == "TIMEOUT":
        print(f"⏱️ [TIMEOUT] {timeout_seconds}초 내에 결론을 내지 못했습니다.")
        return {"status": "TIMEOUT"}

    print(f"⚠️ [UNKNOWN/ERROR] Marabou 종료 코드: {exit_code}")
    return {"status": exit_code.upper()}

# 실행 예시 (가상의 데이터)
np.random.seed(0)
weight_scale = 0.1
dummy_weights = [
    weight_scale * np.random.randn(64, 20),
    weight_scale * np.random.randn(32, 64),
    weight_scale * np.random.randn(1, 32),
]
dummy_biases = [np.random.randn(64), np.random.randn(32), np.random.randn(1)]
verify_monotonicity(dummy_weights, dummy_biases, income_idx=0, input_dim=20, timeout_seconds=60)