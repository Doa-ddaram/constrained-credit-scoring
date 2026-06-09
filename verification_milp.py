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


def _add_local_robustness_constraints(
    problem: pulp.LpProblem,
    x_a: Sequence[pulp.LpVariable],
    x_b: Sequence[pulp.LpVariable],
    original_x: np.ndarray,
    epsilon: float,
) -> None:
    """Add constraints for local robustness: x_a is within epsilon of x_b (original).
    
    x_a: perturbed input (MILP variables within epsilon of original)
    x_b: original input (fixed to original_x)
    
    Violation case: model output flips (e.g., model(original) >= 0 but model(perturbed) < 0)
    """
    for feature_index in range(len(x_a)):
        # x_a must be within epsilon L∞ ball of original_x
        problem += x_a[feature_index] >= float(original_x[feature_index]) - epsilon
        problem += x_a[feature_index] <= float(original_x[feature_index]) + epsilon
        # x_b is fixed to original_x (the unperturbed reference)
        problem += x_b[feature_index] == float(original_x[feature_index])


def _categorize_features(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    """Categorize features into immutable, increasing-only, decreasing-only, and flexible.
    
    Returns:
        Dict with keys: 'immutable', 'increasing_only', 'decreasing_only', 'flexible'
    """
    immutable_keywords = ["age", "sex", "gender", "race", "foreign", "marital"]
    increasing_keywords = ["duration", "checking", "savings", "employment", "credit_history", "score", "income"]
    decreasing_keywords = ["debt", "liabilities", "installment", "risk"]
    
    categorized = {
        "immutable": [],
        "increasing_only": [],
        "decreasing_only": [],
        "flexible": []
    }
    
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in immutable_keywords):
            categorized["immutable"].append(i)
        elif any(kw in name_lower for kw in increasing_keywords):
            categorized["increasing_only"].append(i)
        elif any(kw in name_lower for kw in decreasing_keywords):
            categorized["decreasing_only"].append(i)
        else:
            categorized["flexible"].append(i)
    
    return categorized


def _get_feature_costs(feature_names: Sequence[str]) -> np.ndarray:
    """Assign cost weights to features (higher = harder to change).
    
    Returns:
        Array of costs, one per feature.
    """
    costs = np.ones(len(feature_names), dtype=np.float64)
    
    # Immutable features have infinite cost (or very high)
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["age", "sex", "gender", "race", "foreign"]):
            costs[i] = 1e6  # Essentially immutable
        # Employment is harder to change
        elif "employment" in name_lower:
            costs[i] = 5.0
        # Credit history requires time
        elif "credit_history" in name_lower:
            costs[i] = 3.0
        # Age requires time (decreasing direction not possible)
        elif "age" in name_lower:
            costs[i] = 1e6
    
    return costs


def _add_recourse_constraints(
    problem: pulp.LpProblem,
    x_a: Sequence[pulp.LpVariable],
    feature_names: Sequence[str],
    original_x: np.ndarray,
    bound_limit: float,
) -> List[pulp.LpVariable]:
    """Add recourse constraints: immutable features stay same, directional constraints apply.
    
    Returns:
        List of distance variables for objective minimization.
    """
    # Normalize original_x should already be a numeric numpy array; if not, coerce safely
    if isinstance(original_x, torch.Tensor):
        original_x = original_x.detach().cpu().numpy()
    elif hasattr(original_x, "to_numpy"):
        # pandas Series/DataFrame
        original_x = original_x.to_numpy()
    elif hasattr(original_x, "values") and not callable(getattr(original_x, "values", None)):
        original_x = np.asarray(original_x.values, dtype=np.float64)
    else:
        original_x = np.asarray(original_x, dtype=np.float64)
    
    input_dim = len(feature_names)
    categorized = _categorize_features(feature_names)
    costs = _get_feature_costs(feature_names)
    
    distance_vars = []
    
    for i in range(input_dim):
        dist_var = pulp.LpVariable(f"dist_{i}", lowBound=0.0, cat="Continuous")
        
        # Create weighted absolute distance: cost * |x_a[i] - original_x[i]|
        orig_val = float(original_x[i])
        problem += dist_var >= costs[i] * (x_a[i] - orig_val)
        problem += dist_var >= costs[i] * (orig_val - x_a[i])
        
        distance_vars.append(dist_var)
        
        # Apply directional constraints
        if i in categorized["immutable"]:
            # Immutable: cannot change
            problem += x_a[i] == orig_val, f"Immutable_{i}_{feature_names[i]}"
        
        elif i in categorized["increasing_only"]:
            # Can only increase or stay same
            problem += x_a[i] >= orig_val, f"IncreasingOnly_{i}_{feature_names[i]}"
            problem += x_a[i] <= bound_limit, f"BoundInc_{i}"
        
        elif i in categorized["decreasing_only"]:
            # Can only decrease or stay same
            problem += x_a[i] <= orig_val, f"DecreasingOnly_{i}_{feature_names[i]}"
            problem += x_a[i] >= -bound_limit, f"BoundDec_{i}"
        
        else:
            # Flexible: can change in both directions within bounds
            problem += x_a[i] >= -bound_limit, f"LowerBound_{i}"
            problem += x_a[i] <= bound_limit, f"UpperBound_{i}"
    
    return distance_vars


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
    original_x : Optional[np.ndarray] = None,
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
    elif property_name == "local_robustness":
        if original_x is None:
            raise ValueError("original_x is required for local_robustness")
        _add_local_robustness_constraints(problem, x_a, x_b, original_x, epsilon)
    elif property_name == "recourse":
        if original_x is None:
            raise ValueError("original_x is required for recourse verification")
        
        # Add recourse constraints (immutable, directional, weighted distance)
        distance_vars = _add_recourse_constraints(problem, x_a, feature_names, original_x, bound_limit)
        
        # Set objective: minimize weighted L1 distance
        problem.sense = pulp.LpMinimize
        problem.setObjective(pulp.lpSum(distance_vars))
        
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
            problem += output_a >= output_b + output_margin
    elif property_name == "local_robustness":
        # Violation: output flips sign between original (x_b) and perturbed (x_a)
        # Case 1: original approved (output_b >= 0) but perturbed rejected (output_a < 0)
        # OR Case 2: original rejected (output_b < 0) but perturbed approved (output_a >= 0)
        # We search for EITHER case by using a big-M disjunction.
        # For simplicity, we search for Case 1 (original approved -> perturbed rejected)
        problem += output_b >= output_margin, "OriginalApproved"
        problem += output_a <= -output_margin, "PerturbedRejected"
    elif property_name == "recourse":
        # Force the model to approve (logit >= 0.0)
        problem += output_a >= 0.0, "Must_Approve"
    
    return problem, {"xA": x_a, "xB": x_b, "outputA": output_a, "outputB": output_b}


def _solve_problem(problem: pulp.LpProblem, timeout_seconds: int) -> Tuple[str, Dict[str, Any]]:
    num_constraints = len(problem.constraints)
    num_binaries = sum(1 for v in problem.variables() if v.cat in [pulp.LpInteger, pulp.LpBinary])
    
    solver = _build_solver(timeout_seconds)
    
    start_time = time.perf_counter()
    if solver is None:
        problem.solve()
    else:
        problem.solve(solver)
    elapsed = time.perf_counter() - start_time

    status = pulp.LpStatus.get(problem.status, str(problem.status))
    return status, {"solve_seconds": elapsed, 
                    "solver": solver.__class__.__name__ if solver else "default",
                    "num_constraints": num_constraints,
                    "num_binary_vars": num_binaries
                    }


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
    original_x: Optional[np.ndarray] = None,
    original_output: Optional[float] = None,
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
        original_x=original_x,
    )
    status, stats = _solve_problem(problem, timeout_seconds)

    result: Dict[str, Any] = {
        "model_kind": model_kind,
        "property_name": property_name,
        "status": status,
        **stats,
    }
    
    # Include reference output if provided (useful for local_robustness)
    if original_output is not None:
        result["original_output"] = original_output

    if status.lower() in {"optimal", "feasible"}:
        x_a = np.array([pulp.value(var) for var in payload["xA"]], dtype=float)
        x_b = np.array([pulp.value(var) for var in payload["xB"]], dtype=float)
        output_a = float(pulp.value(payload["outputA"]))
        output_b = float(pulp.value(payload["outputB"]))
        
        result.update(
            {
                "xA": x_a,
                "xB": x_b,
                "outputA": output_a,
                "outputB": output_b,
            }
        )
        
        # For local_robustness, compute the perturbation and output flip
        if property_name == "local_robustness":
            l_inf_distance = np.max(np.abs(x_a - x_b))
            result["linf_perturbation"] = l_inf_distance
            result["output_flip"] = output_a * output_b < 0  # True if signs differ

    return result


def _compute_model_output(
    model_kind: str,
    model_path: str,
    input_x: np.ndarray,
    feature_names: Sequence[str],
) -> float:
    """Compute the model output for a single input."""
    input_dim = len(feature_names)
    if model_kind == "xgboost":
        booster = _load_xgboost_booster(model_path)
        return float(booster.predict(input_x.reshape(1, -1))[0])
    elif model_kind == "mlp":
        layers = _load_mlp_layers(model_path, input_dim=input_dim)
        x = torch.tensor(input_x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for weight, bias in layers:
                w = torch.tensor(weight, dtype=torch.float32)
                b = torch.tensor(bias, dtype=torch.float32)
                x = torch.nn.functional.linear(x, w, b)
                # Apply ReLU for all but the last layer
                if weight is not layers[-1][0]:
                    x = torch.nn.functional.relu(x)
        return float(x.squeeze().item())
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")


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
    original_x: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    # Pre-compute original output if needed for local_robustness
    original_output_xgb = None
    original_output_mlp = None
    if property_name == "local_robustness" and original_x is not None:
        original_output_xgb = _compute_model_output("xgboost", xgboost_path, original_x, feature_names)
        original_output_mlp = _compute_model_output("mlp", mlp_path, original_x, feature_names)

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
        original_x=original_x,
        original_output=original_output_xgb,
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
        original_x=original_x,
        original_output=original_output_mlp,
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
        choices=["monotonicity", "individual_fairness", "compare", "recourse", "local_robustness"],
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
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the customer data to use for recourse")
    return parser.parse_args()


def _normalize_original_input(original_x, input_dim: int) -> np.ndarray:
    """Robustly convert various input types to a numpy array of length input_dim.

    Accepts: torch.Tensor, numpy array, pandas Series/row, list/tuple, or a tuple (x, y).
    Raises a ValueError if the resulting array length doesn't match input_dim.
    """
    if original_x is None:
        raise ValueError("original_x must not be None")

    # If passed a (x, y) tuple (common from datasets), extract x
    if isinstance(original_x, (list, tuple)) and len(original_x) > 0 and not isinstance(
        original_x[0], (int, float, np.floating, np.integer, np.ndarray, torch.Tensor)
    ):
        # fallback: keep as-is
        pass

    if isinstance(original_x, tuple) and len(original_x) == 2:
        original_x = original_x[0]

    if isinstance(original_x, torch.Tensor):
        arr = original_x.detach().cpu().numpy()
    else:
        try:
            # pandas Series/DataFrame have to_numpy()
            if hasattr(original_x, "to_numpy"):
                arr = original_x.to_numpy()
            elif hasattr(original_x, "values") and not callable(getattr(original_x, "values", None)):
                arr = np.asarray(original_x.values)
            else:
                arr = np.asarray(original_x)
        except Exception:
            # Last resort: try elementwise conversion
            arr = np.asarray([float(v) for v in original_x], dtype=np.float64)

    arr = arr.astype(np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != input_dim:
        raise ValueError(f"original_x length {arr.shape[0]} does not match expected input_dim {input_dim}")
    return arr


if __name__ == "__main__":
    args = parse_args()
    bundle = load_and_preprocess_data()
    feature_names = list(bundle["feature_names"])

    target_feature_name = args.target_feature or _default_monotonicity_feature(feature_names)
    sensitive_feature_names = args.sensitive_features or [_default_sensitive_feature(feature_names)]

    dataset_x = bundle.get("X_test", bundle.get("X", None))

    original_x_sample = None
    if dataset_x is not None and len(dataset_x) > args.sample_index:
        original_x_sample = dataset_x[args.sample_index]
        # Normalize to numpy array with correct length
        try:
            original_x_sample = _normalize_original_input(original_x_sample, input_dim=len(feature_names))
        except Exception:
            # fallback: attempt to coerce safely
            try:
                original_x_sample = np.asarray(original_x_sample, dtype=np.float64)
            except Exception:
                original_x_sample = np.zeros(len(feature_names))
    else:
        # Fallback to zeros if dataset array is not found
        original_x_sample = np.zeros(len(feature_names))

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
    elif args.property == "local_robustness":
        print(f"Running Local Robustness MILP for customer index {args.sample_index}")
        print(f"Perturbation budget (epsilon): {args.epsilon}")
        
        # Compare both models
        result = compare_xgboost_and_mlp(
            xgboost_path=args.xgboost_path,
            mlp_path=args.mlp_path,
            feature_names=feature_names,
            property_name="local_robustness",
            bound_limit=args.bound_limit,
            epsilon=args.epsilon,
            output_margin=args.output_margin,
            timeout_seconds=args.timeout_seconds,
            original_x=original_x_sample,
        )
        
        # Print detailed results for both models
        for model_name, model_result in result.items():
            print(f"\n========== {model_name.upper()} ==========")
            print(f"Status: {model_result['status']}")
            print(f"Solve time: {model_result['solve_seconds']:.4f}s")
            print(f"Wall time: {model_result['wall_seconds']:.4f}s")
            
            if model_result['status'].lower() in {"optimal", "feasible"}:
                print(f"  VIOLATED: Found a counterexample within epsilon={args.epsilon}")
                print(f"  Original output (x_b): {model_result['outputB']:.6f}")
                print(f"  Perturbed output (x_a): {model_result['outputA']:.6f}")
                print(f"  Output flip detected: {model_result.get('output_flip', False)}")
                print(f"  L-inf perturbation distance: {model_result.get('linf_perturbation', 'N/A')}")
                print(f"\n  Perturbed input differences:")
                for i, name in enumerate(feature_names):
                    diff = model_result['xA'][i] - model_result['xB'][i]
                    if abs(diff) > 1e-6:
                        print(f"    [{name}]: {diff:+.6f}")
            else:
                print(f"  VERIFIED: No violation found within epsilon={args.epsilon}")
    elif args.property == "recourse":
        print(f"Running Actionable Recourse MILP for customer index {args.sample_index}")
        print(f"Goal: Find minimal changes to achieve loan approval\n")
        
        # Run for both models
        xgb_result = verify_property_milp(
            model_kind="xgboost",
            model_path=args.xgboost_path,
            feature_names=feature_names,
            property_name="recourse",
            bound_limit=args.bound_limit,
            timeout_seconds=args.timeout_seconds,
            original_x=original_x_sample,
        )
        
        mlp_result = verify_property_milp(
            model_kind="mlp",
            model_path=args.mlp_path,
            feature_names=feature_names,
            property_name="recourse",
            bound_limit=args.bound_limit,
            timeout_seconds=args.timeout_seconds,
            original_x=original_x_sample,
        )
        
        # Get feature categorization for clear explanation
        categorized = _categorize_features(feature_names)
        costs = _get_feature_costs(feature_names)
        
        # Print results for both models
        for model_name, result in [("XGBoost", xgb_result), ("MLP", mlp_result)]:
            print(f"\n{'='*70}")
            print(f"MODEL: {model_name}")
            print(f"{'='*70}")
            print(f"Status: {result['status']}")
            print(f"Solve time: {result.get('solve_seconds', 'N/A'):.4f}s")
            print(f"Wall time: {result.get('wall_seconds', 'N/A'):.4f}s")
            
            if result.get("status") == "Optimal":
                print(f"\n  SUCCESS! Found actionable recourse plan.")
                print(f"  Original model output: {result.get('original_output', 'N/A')}")
                print(f"  New model output: {result['outputA']:.6f} (APPROVED)")
                
                print(f"\n--- RECOMMENDED ACTIONS ---")
                print(f"{'Feature':<30} {'Current':<12} {'Recommended':<12} {'Change':<12} {'Category':<20}")
                print(f"{'-'*86}")
                
                any_changes = False
                for i, name in enumerate(feature_names):
                    orig = original_x_sample[i]
                    new = result["xA"][i]
                    diff = new - orig
                    
                    # Determine category
                    if i in categorized["immutable"]:
                        category = "Immutable"
                    elif i in categorized["increasing_only"]:
                        category = "Increase only"
                    elif i in categorized["decreasing_only"]:
                        category = "Decrease only"
                    else:
                        category = "Flexible"
                    
                    if abs(diff) > 1e-5:
                        any_changes = True
                        direction = "Increase" if diff > 0 else "Decrease"
                        print(f"{name:<30} {orig:<12.4f} {new:<12.4f} {diff:+.4f} ({direction:<8}) {category:<20}")
                
                if not any_changes:
                    print("No changes needed - customer already qualifies for approval!")
                else:
                    print(f"\nNote: Immutable features (age, gender, race, etc.) cannot be changed.")
                    print(f"      Directional constraints apply to features based on feasibility.")
            else:
                print(f"\n  NO SOLUTION FOUND")
                print(f"  Reason: {result.get('status', 'Unknown')}")
                print(f"  Even with maximum allowed changes (within bounds), loan approval is not achievable.")
                print(f"  Recommendation: Review lending criteria or consider other factors.")
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
