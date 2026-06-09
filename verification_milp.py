import argparse
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pulp
import torch
import torch.nn as nn
import xgboost as xgb

from dataset import load_and_preprocess_data


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def _normalize_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    alternate = None
    if "saved_weights" in model_path:
        alternate = model_path.replace("saved_weights", "saved_weight")
    elif "saved_weight" in model_path:
        alternate = model_path.replace("saved_weight", "saved_weights")

    if alternate and os.path.exists(alternate):
        return alternate

    return model_path


def _resolve_feature_index(feature_names: Sequence[str], query: str) -> int:
    if query is None:
        raise ValueError("feature query must not be None")

    if query.isdigit():
        index = int(query)
        if index < 0 or index >= len(feature_names):
            raise IndexError(f"feature index out of range: {index}")
        return index

    exact_matches = [i for i, name in enumerate(feature_names) if name == query]
    if exact_matches:
        return exact_matches[0]

    suffix_matches = [i for i, name in enumerate(feature_names) if name.endswith(query)]
    if suffix_matches:
        return suffix_matches[0]

    substring_matches = [i for i, name in enumerate(feature_names) if query in name]
    if substring_matches:
        return substring_matches[0]

    raise ValueError(f"feature '{query}' was not found")


def _resolve_feature_indices(feature_names: Sequence[str], queries: Optional[Sequence[str]]) -> List[int]:
    if not queries:
        return []
    return [_resolve_feature_index(feature_names, query) for query in queries]


def _list_available_solvers() -> List[str]:
    try:
        return pulp.listSolvers(onlyAvailable=True)
    except Exception:
        return []


def _build_solver(timeout_seconds: int):
    available = set(_list_available_solvers())
    if "HiGHS" in available:
        try:
            return pulp.getSolver("HiGHS", timeLimit=timeout_seconds, msg=False)
        except Exception:
            pass
    if "PULP_CBC_CMD" in available:
        return pulp.PULP_CBC_CMD(timeLimit=timeout_seconds, msg=False)
    return None


def _load_xgboost_booster(model_path: str) -> xgb.Booster:
    model_path = _normalize_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model file not found: {model_path}")

    classifier = xgb.XGBClassifier()
    classifier.load_model(model_path)
    return classifier.get_booster()


def _load_mlp_layers(model_path: str, input_dim: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    model_path = _normalize_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MLP model file not found: {model_path}")

    model = BaselineMLP(input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy().astype(np.float64)
            bias = module.bias.detach().cpu().numpy().astype(np.float64)
            layers.append((weight, bias))

    if not layers:
        raise ValueError("no Linear layers were found in the loaded MLP")

    return layers


def _parse_xgboost_trees(booster: xgb.Booster) -> List[Dict[str, Any]]:
    df = booster.trees_to_dataframe()
    trees: List[Dict[str, Any]] = []

    for tree_id in sorted(df["Tree"].unique()):
        tree_df = df[df["Tree"] == tree_id].copy()
        nodes = {int(row["Node"]): row.to_dict() for _, row in tree_df.iterrows()}

        def dfs(node_id: int, path: List[Tuple[int, float, str]]) -> List[Dict[str, Any]]:
            row = nodes[node_id]
            feature = str(row.get("Feature", ""))
            if feature.lower() == "leaf":
                leaf_value = float(row.get("Gain", 0.0))
                return [
                    {
                        "leaf_id": node_id,
                        "value": leaf_value,
                        "path": list(path),
                    }
                ]

            if not feature.startswith("f"):
                raise ValueError(f"unexpected XGBoost feature label: {feature}")

            feature_index = int(feature[1:])
            split = float(row["Split"])

            def parse_child(value: Any) -> int:
                if isinstance(value, str) and "-" in value:
                    return int(value.split("-")[-1])
                return int(value)

            yes_node = parse_child(row["Yes"])
            no_node = parse_child(row["No"])

            leaves: List[Dict[str, Any]] = []
            leaves.extend(dfs(yes_node, path + [(feature_index, split, "le")]))
            leaves.extend(dfs(no_node, path + [(feature_index, split, "gt")]))
            return leaves

        trees.append({"tree_id": int(tree_id), "leaves": dfs(0, [])})

    return trees


def _interval_affine_bounds(
    weight: np.ndarray,
    bias: np.ndarray,
    prev_lower: np.ndarray,
    prev_upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.empty(weight.shape[0], dtype=np.float64)
    upper = np.empty(weight.shape[0], dtype=np.float64)

    for row_index in range(weight.shape[0]):
        row = weight[row_index]
        row_lower = float(bias[row_index])
        row_upper = float(bias[row_index])
        for coeff, low, high in zip(row, prev_lower, prev_upper):
            if coeff >= 0:
                row_lower += coeff * low
                row_upper += coeff * high
            else:
                row_lower += coeff * high
                row_upper += coeff * low
        lower[row_index] = row_lower
        upper[row_index] = row_upper

    return lower, upper


def _encode_mlp_forward_pass(
    problem: pulp.LpProblem,
    input_vars: Sequence[pulp.LpVariable],
    layers: Sequence[Tuple[np.ndarray, np.ndarray]],
    prefix: str,
    input_bound: float,
) -> pulp.LpAffineExpression:
    prev_vars: List[Any] = list(input_vars)
    prev_lower = np.full(len(prev_vars), -input_bound, dtype=np.float64)
    prev_upper = np.full(len(prev_vars), input_bound, dtype=np.float64)

    for layer_index, (weight, bias) in enumerate(layers):
        if weight.shape[1] != len(prev_vars):
            raise ValueError(
                f"layer {layer_index} input mismatch: expected {len(prev_vars)}, got {weight.shape[1]}"
            )

        pre_lower, pre_upper = _interval_affine_bounds(weight, bias, prev_lower, prev_upper)
        layer_is_output = layer_index == len(layers) - 1
        next_vars: List[Any] = []

        for neuron_index in range(weight.shape[0]):
            z_var = pulp.LpVariable(
                f"{prefix}_z_{layer_index}_{neuron_index}",
                lowBound=float(pre_lower[neuron_index]),
                upBound=float(pre_upper[neuron_index]),
                cat="Continuous",
            )

            affine = pulp.lpSum(
                float(weight[neuron_index, input_index]) * prev_vars[input_index]
                for input_index in range(weight.shape[1])
            ) + float(bias[neuron_index])
            problem += z_var == affine

            if layer_is_output:
                next_vars.append(z_var)
                continue

            lower_bound = float(pre_lower[neuron_index])
            upper_bound = float(pre_upper[neuron_index])

            if upper_bound <= 0.0:
                a_var = pulp.LpVariable(
                    f"{prefix}_a_{layer_index}_{neuron_index}",
                    lowBound=0.0,
                    upBound=0.0,
                    cat="Continuous",
                )
                problem += a_var == 0.0
                next_vars.append(a_var)
                continue

            if lower_bound >= 0.0:
                a_var = pulp.LpVariable(
                    f"{prefix}_a_{layer_index}_{neuron_index}",
                    lowBound=lower_bound,
                    upBound=upper_bound,
                    cat="Continuous",
                )
                problem += a_var == z_var
                next_vars.append(a_var)
                continue

            binary = pulp.LpVariable(f"{prefix}_r_{layer_index}_{neuron_index}", cat="Binary")
            a_var = pulp.LpVariable(
                f"{prefix}_a_{layer_index}_{neuron_index}",
                lowBound=0.0,
                upBound=max(0.0, upper_bound),
                cat="Continuous",
            )
            problem += a_var >= z_var
            problem += a_var >= 0.0
            problem += a_var <= z_var - lower_bound * (1 - binary)
            problem += a_var <= upper_bound * binary
            next_vars.append(a_var)

        prev_vars = next_vars
        prev_lower = np.maximum(0.0, pre_lower) if not layer_is_output else pre_lower
        prev_upper = np.maximum(0.0, pre_upper) if not layer_is_output else pre_upper

    if len(prev_vars) != 1:
        raise ValueError("final layer must produce a single output logit")

    return prev_vars[0]


def _encode_xgboost_forward_pass(
    problem: pulp.LpProblem,
    input_vars: Sequence[pulp.LpVariable],
    trees: Sequence[Dict[str, Any]],
    prefix: str,
    input_bound: float,
) -> pulp.LpAffineExpression:
    big_m = max(1000.0, 4.0 * input_bound + 1.0)
    total_output: List[pulp.LpAffineExpression] = []

    for tree_index, tree in enumerate(trees):
        leaves = tree["leaves"]
        if not leaves:
            raise ValueError(f"tree {tree_index} has no leaves")

        leaf_choice = {
            leaf["leaf_id"]: pulp.LpVariable(f"{prefix}_t{tree_index}_leaf_{leaf['leaf_id']}", cat="Binary")
            for leaf in leaves
        }

        problem += pulp.lpSum(leaf_choice[leaf["leaf_id"]] for leaf in leaves) == 1

        for leaf in leaves:
            leaf_id = leaf["leaf_id"]
            indicator = leaf_choice[leaf_id]
            for feature_index, split, direction in leaf["path"]:
                if direction == "le":
                    problem += input_vars[feature_index] <= split + big_m * (1 - indicator)
                else:
                    problem += input_vars[feature_index] >= split + 1e-8 - big_m * (1 - indicator)

        total_output.append(
            pulp.lpSum(float(leaf["value"]) * leaf_choice[leaf["leaf_id"]] for leaf in leaves)
        )

    return pulp.lpSum(total_output)


def _create_pairwise_inputs(
    problem: pulp.LpProblem,
    input_dim: int,
    bound_limit: float,
) -> Tuple[List[pulp.LpVariable], List[pulp.LpVariable]]:
    x_a = [
        pulp.LpVariable(f"xA_{index}", lowBound=-bound_limit, upBound=bound_limit, cat="Continuous")
        for index in range(input_dim)
    ]
    x_b = [
        pulp.LpVariable(f"xB_{index}", lowBound=-bound_limit, upBound=bound_limit, cat="Continuous")
        for index in range(input_dim)
    ]
    return x_a, x_b


def _add_monotonicity_constraints(
    problem: pulp.LpProblem,
    x_a: Sequence[pulp.LpVariable],
    x_b: Sequence[pulp.LpVariable],
    target_index: int,
    epsilon: float,
) -> None:
    for feature_index in range(len(x_a)):
        if feature_index != target_index:
            problem += x_a[feature_index] == x_b[feature_index]
    problem += x_b[target_index] >= x_a[target_index] + epsilon


def _add_individual_fairness_constraints(
    problem: pulp.LpProblem,
    x_a: Sequence[pulp.LpVariable],
    x_b: Sequence[pulp.LpVariable],
    sensitive_indices: Sequence[int],
) -> None:
    sensitive_set = set(sensitive_indices)
    for feature_index in range(len(x_a)):
        if feature_index not in sensitive_set:
            problem += x_a[feature_index] == x_b[feature_index]


def _build_problem_for_model(
    model_kind: str,
    model_path: str,
    feature_names: Sequence[str],
    property_name: str,
    target_feature_name: Optional[str],
    sensitive_feature_names: Optional[Sequence[str]],
    bound_limit: float,
    epsilon: float,
    output_margin: float,
) -> Tuple[pulp.LpProblem, Dict[str, Any]]:
    input_dim = len(feature_names)
    problem = pulp.LpProblem(f"{model_kind}_{property_name}_verification", pulp.LpMinimize)
    problem += 0

    x_a, x_b = _create_pairwise_inputs(problem, input_dim, bound_limit)

    if property_name == "monotonicity":
        if target_feature_name is None:
            raise ValueError("target_feature_name is required for monotonicity")
        target_index = _resolve_feature_index(feature_names, target_feature_name)
        _add_monotonicity_constraints(problem, x_a, x_b, target_index, epsilon)
    elif property_name == "individual_fairness":
        sensitive_indices = _resolve_feature_indices(feature_names, sensitive_feature_names)
        if not sensitive_indices:
            raise ValueError("sensitive_feature_names is required for individual_fairness")
        _add_individual_fairness_constraints(problem, x_a, x_b, sensitive_indices)
    else:
        raise ValueError(f"unsupported property: {property_name}")

    if model_kind == "xgboost":
        booster = _load_xgboost_booster(model_path)
        model_payload = _parse_xgboost_trees(booster)
        output_a = _encode_xgboost_forward_pass(problem, x_a, model_payload, "A", bound_limit)
        output_b = _encode_xgboost_forward_pass(problem, x_b, model_payload, "B", bound_limit)
    elif model_kind == "mlp":
        layers = _load_mlp_layers(model_path, input_dim=input_dim)
        model_payload = layers
        output_a = _encode_mlp_forward_pass(problem, x_a, model_payload, "A", bound_limit)
        output_b = _encode_mlp_forward_pass(problem, x_b, model_payload, "B", bound_limit)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")

    if property_name == "monotonicity":
        if "duration" in target_feature_name or "amount" in target_feature_name:
          problem += output_b >= output_a + output_margin
        else:
            problem += output_a >= output_b + epsilon

    return problem, {"xA": x_a, "xB": x_b, "outputA": output_a, "outputB": output_b}


def _solve_problem(problem: pulp.LpProblem, timeout_seconds: int) -> Tuple[str, Dict[str, Any]]:
    solver = _build_solver(timeout_seconds)
    start_time = time.perf_counter()
    if solver is None:
        problem.solve()
    else:
        problem.solve(solver)
    elapsed = time.perf_counter() - start_time

    status = pulp.LpStatus.get(problem.status, str(problem.status))
    return status, {"solve_seconds": elapsed, "solver": solver.__class__.__name__ if solver else "default"}


def verify_property_milp(
    model_kind: str,
    model_path: str,
    feature_names: Sequence[str],
    property_name: str,
    target_feature_name: Optional[str] = None,
    sensitive_feature_names: Optional[Sequence[str]] = None,
    bound_limit: float = 3.0,
    epsilon: float = 0.01,
    output_margin: float = 1e-6,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    problem, payload = _build_problem_for_model(
        model_kind=model_kind,
        model_path=model_path,
        feature_names=feature_names,
        property_name=property_name,
        target_feature_name=target_feature_name,
        sensitive_feature_names=sensitive_feature_names,
        bound_limit=bound_limit,
        epsilon=epsilon,
        output_margin=output_margin,
    )
    status, stats = _solve_problem(problem, timeout_seconds)

    result: Dict[str, Any] = {
        "model_kind": model_kind,
        "property_name": property_name,
        "status": status,
        **stats,
    }

    if status.lower() in {"optimal", "feasible"}:
        x_a = np.array([pulp.value(var) for var in payload["xA"]], dtype=float)
        x_b = np.array([pulp.value(var) for var in payload["xB"]], dtype=float)
        result.update(
            {
                "xA": x_a,
                "xB": x_b,
                "outputA": float(pulp.value(payload["outputA"])),
                "outputB": float(pulp.value(payload["outputB"])),
            }
        )

    return result


def compare_xgboost_and_mlp(
    xgboost_path: str,
    mlp_path: str,
    feature_names: Sequence[str],
    property_name: str,
    target_feature_name: Optional[str] = None,
    sensitive_feature_names: Optional[Sequence[str]] = None,
    bound_limit: float = 3.0,
    epsilon: float = 0.01,
    output_margin: float = 1e-6,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    xgb_start = time.perf_counter()
    xgb_result = verify_property_milp(
        model_kind="xgboost",
        model_path=xgboost_path,
        feature_names=feature_names,
        property_name=property_name,
        target_feature_name=target_feature_name,
        sensitive_feature_names=sensitive_feature_names,
        bound_limit=bound_limit,
        epsilon=epsilon,
        output_margin=output_margin,
        timeout_seconds=timeout_seconds,
    )
    xgb_result["wall_seconds"] = time.perf_counter() - xgb_start

    mlp_start = time.perf_counter()
    mlp_result = verify_property_milp(
        model_kind="mlp",
        model_path=mlp_path,
        feature_names=feature_names,
        property_name=property_name,
        target_feature_name=target_feature_name,
        sensitive_feature_names=sensitive_feature_names,
        bound_limit=bound_limit,
        epsilon=epsilon,
        output_margin=output_margin,
        timeout_seconds=timeout_seconds,
    )
    mlp_result["wall_seconds"] = time.perf_counter() - mlp_start

    return {"xgboost": xgb_result, "mlp": mlp_result}


def _default_monotonicity_feature(feature_names: Sequence[str]) -> str:
    candidates = ["num__duration", "num__credit_amount", "duration", "credit_amount"]
    for candidate in candidates:
        for name in feature_names:
            if candidate in name:
                return name
    for name in feature_names:
        if "duration" in name.lower() or "amount" in name.lower():
            return name
    return feature_names[0]


def _default_sensitive_feature(feature_names: Sequence[str]) -> str:
    candidates = ["num__age", "age"]
    for candidate in candidates:
        for name in feature_names:
            if candidate in name:
                return name
    return feature_names[0]

def analyze_milp_complexity(prob: pulp.LpProblem, solve_time: float) -> Dict[str, Any]:
    """
    Extract exact mathematical complexity and results from the PuLP problem.
    """
    # Calculate the total number of constraints in the problem
    num_constraints = len(prob.constraints)
    
    # Calculate total variables and specifically binary variables
    variables = prob.variables()
    num_total_vars = len(variables)
    
    # Binary variables are the main bottleneck for NP-Hard MILP problems
    num_binaries = sum(
        1 for v in variables 
        if v.cat == pulp.LpInteger or v.cat == pulp.LpBinary
    )
    
    # Extract objective value if optimal
    status = pulp.LpStatus[prob.status]
    objective_value = None
    if status == 'Optimal':
        # Retrieve the maximum violation margin or minimum distance
        objective_value = pulp.value(prob.objective)
        
    return {
        "status": status,
        "solve_time_sec": round(solve_time, 4),
        "objective_value": objective_value,
        "num_constraints": num_constraints,
        "num_total_vars": num_total_vars,
        "num_binary_vars": num_binaries
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MILP verification for XGBoost and baseline MLP")
    parser.add_argument(
        "--property",
        choices=["monotonicity", "individual_fairness", "compare"],
        default="compare",
        help="property to verify",
    )
    parser.add_argument("--xgboost-path", type=str, default="saved_weights/best_xgboost.json")
    parser.add_argument("--mlp-path", type=str, default="saved_weights/best_baseline_mlp.pth")
    parser.add_argument("--target-feature", type=str, default=None)
    parser.add_argument("--sensitive-features", type=str, nargs="*", default=None)
    parser.add_argument("--bound-limit", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--output-margin", type=float, default=1e-6)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle = load_and_preprocess_data()
    feature_names = list(bundle["feature_names"])

    target_feature_name = args.target_feature or _default_monotonicity_feature(feature_names)
    sensitive_feature_names = args.sensitive_features or [_default_sensitive_feature(feature_names)]

    if args.property == "compare":
        print(f"Comparing XGBoost and MLP with monotonicity over '{target_feature_name}'")
        result = compare_xgboost_and_mlp(
            xgboost_path=args.xgboost_path,
            mlp_path=args.mlp_path,
            feature_names=feature_names,
            property_name="monotonicity",
            target_feature_name=target_feature_name,
            bound_limit=args.bound_limit,
            epsilon=args.epsilon,
            output_margin=args.output_margin,
            timeout_seconds=args.timeout_seconds,
        )
        print(result)
    elif args.property == "monotonicity":
        print(f"Running monotonicity MILP over '{target_feature_name}'")
        print(
            verify_property_milp(
                model_kind="mlp",
                model_path=args.mlp_path,
                feature_names=feature_names,
                property_name="monotonicity",
                target_feature_name=target_feature_name,
                bound_limit=args.bound_limit,
                epsilon=args.epsilon,
                output_margin=args.output_margin,
                timeout_seconds=args.timeout_seconds,
            )
        )
    else:
        print(f"Running individual fairness MILP over sensitive feature(s) {sensitive_feature_names}")
        print(
            verify_property_milp(
                model_kind="mlp",
                model_path=args.mlp_path,
                feature_names=feature_names,
                property_name="individual_fairness",
                sensitive_feature_names=sensitive_feature_names,
                bound_limit=args.bound_limit,
                epsilon=args.epsilon,
                output_margin=args.output_margin,
                timeout_seconds=args.timeout_seconds,
            )
        )
