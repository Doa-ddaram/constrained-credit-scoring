from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import numpy as np

def verify_individual_fairness(weights, biases, sensitive_indices, input_dim, epsilon=0.1):
    """
    개별적 공정성(Individual Fairness) 제약 위반 여부를 SMT 솔버로 검증하는 함수
    
    :param weights: 모델 가중치 리스트
    :param biases: 모델 편향 리스트
    :param sensitive_indices: 성별, 연령 등 민감 속성(Sensitive attributes)의 인덱스 리스트
    :param input_dim: 전체 피처 개수
    :param epsilon: 허용되는 최대 로짓(Logit) 차이 (이 값을 넘어가면 편향된 것으로 간주)
    """
    print(f"🔍 [공정성 검증] 허용 오차(epsilon) {epsilon} 기준으로 Marabou 탐색 시작...")
    network = MarabouNetwork()


    # 1. 두 명의 가상 지원자 A, B 생성
    x_A = [network.getNewVariable() for _ in range(input_dim)]
    x_B = [network.getNewVariable() for _ in range(input_dim)]

    # 2. 탐색 바운드 설정 (연산 효율을 위해 정규화된 범위로 제한)
    bound_limit = 3.0
    for i in range(input_dim):
        network.setLowerBound(x_A[i], -bound_limit)
        network.setUpperBound(x_A[i], bound_limit)
        network.setLowerBound(x_B[i], -bound_limit)
        network.setUpperBound(x_B[i], bound_limit)

    # 3. 네트워크 순전파 연산 구축 (단조성 코드와 동일한 로직 사용)
    def build_forward_pass(input_vars):
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

            # Marabou는 유한한 bound가 필요하므로 중간 변수를 넓은 범위로 제한합니다.
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

    logit_A = build_forward_pass(x_A)
    logit_B = build_forward_pass(x_B)

    # 4. 제약 조건 (Property) 인코딩
    
    # 조건 4-1: 민감 속성(sensitive_indices)을 제외한 나머지 모든 스펙은 완전히 동일하다.
    for i in range(input_dim):
        if i not in sensitive_indices:
            eq_same = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            eq_same.addAddend(1, x_A[i])
            eq_same.addAddend(-1, x_B[i])
            eq_same.setScalar(0)
            network.addEquation(eq_same)
            
    # 참고: 민감 속성(예: 성별)에 대해서는 아무런 제약을 걸지 않음으로써, 
    # 솔버가 바운드 내에서 가장 편향이 극대화되는 값을 자유롭게 탐색하도록 둡니다.

    # 조건 4-2 (위반 조건): 두 사람의 대출 심사 결과(Logit) 차이가 epsilon보다 크다!
    # 수학적으로 |logit_A - logit_B| > epsilon 이지만, A와 B는 대칭적인 변수이므로 
    # logit_A - logit_B > epsilon 방향만 검사해도 모든 반례를 찾을 수 있습니다.
    eq_violation = MarabouUtils.Equation(MarabouCore.Equation.GE)
    eq_violation.addAddend(1, logit_A)
    eq_violation.addAddend(-1, logit_B)
    eq_violation.setScalar(epsilon)
    network.addEquation(eq_violation)

    # 5. 솔버 실행
    print("🧠 Marabou가 편향된 의사결정 반례를 찾고 있습니다...")
    options = Marabou.createOptions(verbosity=0)
    exit_code, vals, stats = network.solve(options=options)

    # 6. 결과 반환
    if exit_code == "sat" and len(vals) > 0:
        print(f"🚨 [SAT] 공정성 위반! 스펙은 같으나 민감 속성 때문에 결과가 {epsilon} 이상 벌어지는 반례 발견.")
        return {"status": "SAT"}
    elif exit_code == "unsat":
        print("✅ [UNSAT] 해당 모델은 설정된 바운드 내에서 개별적 공정성을 만족합니다.")
        return {"status": "UNSAT"}
    else:
        print(f"⚠️ [UNKNOWN/ERROR] Marabou 종료 코드: {exit_code}")
    return {"status": exit_code.upper()}

dummy_weights = [np.random.randn(64, 20), np.random.randn(32, 64), np.random.randn(1, 32)]
dummy_biases = [np.random.randn(64), np.random.randn(32), np.random.randn(1)]
verify_individual_fairness(dummy_weights, dummy_biases, sensitive_indices=[0], input_dim=20, epsilon=0.1)