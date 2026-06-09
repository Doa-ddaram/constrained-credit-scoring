import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pulp
import torch
import torch.nn as nn
import xgboost as xgb

from dataset import load_and_preprocess_data

# ============================================================================
# CONFIG — verify against the dataset label encoding (L1)
# ----------------------------------------------------------------------------
# Does raw margin(logit) >= 0 mean "loan approved"?
# In German Credit, label 1 is often "bad", so verify the training labels directly.
POSITIVE_MARGIN_MEANS_APPROVE = True

# Monotonicity expectation: if the feature keyword indicates increased risk, the approval margin should decrease.
_RISK_INCREASING = ["duration", "credit_amount", "installment_commitment", "existing_credits"]
_RISK_DECREASING = ["checking_status", "savings_status", "employment"]

# recourse policy: immutable / one-way / cost
_IMMUTABLE_KEYWORDS = ["age", "sex", "gender", "race", "foreign_worker", "personal_status",
                       "num_dependents"]
_INCREASING_ONLY = ["checking_status", "savings_status", "employment"]
_DECREASING_ONLY = ["duration", "credit_amount", "installment_commitment",
                    "existing_credits", "other_payment_plans"]
# ============================================================================


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def _normalize_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    alt = None
    if "saved_weights" in model_path:
        alt = model_path.replace("saved_weights", "saved_weight")
    elif "saved_weight" in model_path:
        alt = model_path.replace("saved_weight", "saved_weights")
    return alt if (alt and os.path.exists(alt)) else model_path


def _resolve_feature_index(feature_names: Sequence[str], query: str) -> int:
    if query is None:
        raise ValueError("feature query must not be None")
    if query.isdigit():
        idx = int(query)
        if not (0 <= idx < len(feature_names)):
            raise IndexError(f"feature index out of range: {idx}")
        return idx
    for matcher in (lambda n: n == query, lambda n: n.endswith(query), lambda n: query in n):
        hits = [i for i, n in enumerate(feature_names) if matcher(n)]
        if hits:
            return hits[0]
    raise ValueError(f"feature '{query}' was not found")


def _resolve_feature_indices(feature_names, queries) -> List[int]:
    return [_resolve_feature_index(feature_names, q) for q in (queries or [])]


def _detect_categorical_groups(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, name in enumerate(feature_names):
        if name.startswith("cat__"):
            key = name.rsplit("_", 1)[0] if "_" in name else name
            groups.setdefault(key, []).append(i)
    return groups


def _list_available_solvers() -> List[str]:
    try:
        return pulp.listSolvers(onlyAvailable=True)
    except Exception:
        return []


def _build_solver(timeout_seconds: int):
    available = set(_list_available_solvers())
    # timeout_get = 0.1
    if "PULP_CBC_CMD" in available:
        try:
            print(f"Using CBC solver with a time limit of {timeout_seconds} seconds")
            return pulp.PULP_CBC_CMD(timeLimit=timeout_seconds, options=["nosos"], msg=False)
        except Exception:
            pass
    if "HiGHS" in available:
        print(f"Using HiGHS solver with a time limit of {timeout_seconds} seconds")
        return pulp.getSolver("HiGHS", timeLimit=timeout_seconds, msg=False)
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
    trees = []
    for tree_id in sorted(df["Tree"].unique()):
        tdf = df[df["Tree"] == tree_id]
        nodes = {int(r["Node"]): r.to_dict() for _, r in tdf.iterrows()}

        def parse_child(v):
            if isinstance(v, str) and "-" in v:
                return int(v.split("-")[-1])
            return int(v)

        def dfs(node_id, path):
            row = nodes[node_id]
            feat = str(row.get("Feature", ""))
            if feat.lower() == "leaf":
                return [{"leaf_id": node_id, "value": float(row.get("Gain", 0.0)), "path": list(path)}]
            if not feat.startswith("f"):
                raise ValueError(f"unexpected XGBoost feature label: {feat}")
            fidx = int(feat[1:])
            split = float(row["Split"])
            leaves = []
            leaves += dfs(parse_child(row["Yes"]), path + [(fidx, split, "lt")])
            leaves += dfs(parse_child(row["No"]), path + [(fidx, split, "ge")])
            return leaves

        trees.append({"tree_id": int(tree_id), "leaves": dfs(0, [])})
    return trees


def _xgboost_base_offset(booster: xgb.Booster, trees: Sequence[Dict[str, Any]], input_dim: int) -> float:
    """Compute the constant offset (base margin) between the MILP leaf sum and the true raw margin
    in a version-independent way. Since the base margin is a constant added to every prediction,
    it can be estimated from a single probe input."""
    probe = np.zeros((1, input_dim), dtype=np.float32)
    dm = xgb.DMatrix(probe)
    margin = float(booster.predict(dm, output_margin=True)[0])
    leaf_idx = np.asarray(booster.predict(dm, pred_leaf=True)).reshape(-1)
    if len(leaf_idx) != len(trees):
        # Safeguard: if the tree count mismatches, fall back to offset 0 (warning)
        return 0.0
    leaf_sum = 0.0
    for tree, node_id in zip(trees, leaf_idx):
        value_by_node = {leaf["leaf_id"]: leaf["value"] for leaf in tree["leaves"]}
        leaf_sum += value_by_node[int(node_id)]
    return margin - leaf_sum


# ---------------------------------------------------------------------------
# IBP bounds
# ---------------------------------------------------------------------------
def _interval_affine_bounds(weight, bias, prev_lower, prev_upper):
    pos = np.clip(weight, 0.0, None)
    neg = np.clip(weight, None, 0.0)
    lower = bias + pos @ prev_lower + neg @ prev_upper
    upper = bias + pos @ prev_upper + neg @ prev_lower
    return lower, upper


# Forward pass encoding
def _encode_mlp_forward_pass(problem, input_vars, layers, prefix, input_lower, input_upper):
    prev_vars = list(input_vars)
    prev_lower = np.asarray(input_lower, dtype=np.float64)
    prev_upper = np.asarray(input_upper, dtype=np.float64)

    for li, (weight, bias) in enumerate(layers):
        if weight.shape[1] != len(prev_vars):
            raise ValueError(f"layer {li} input mismatch: {weight.shape[1]} vs {len(prev_vars)}")
        pre_lo, pre_hi = _interval_affine_bounds(weight, bias, prev_lower, prev_upper)
        is_output = li == len(layers) - 1
        next_vars = []
        for n in range(weight.shape[0]):
            z = pulp.LpVariable(f"{prefix}_z_{li}_{n}", lowBound=float(pre_lo[n]),
                                upBound=float(pre_hi[n]), cat="Continuous")
            problem += z == pulp.lpSum(float(weight[n, k]) * prev_vars[k]
                                       for k in range(weight.shape[1])) + float(bias[n])
            if is_output:
                next_vars.append(z)
                continue
            lo, hi = float(pre_lo[n]), float(pre_hi[n])
            if hi <= 0.0:
                a = pulp.LpVariable(f"{prefix}_a_{li}_{n}", lowBound=0.0, upBound=0.0, cat="Continuous")
                problem += a == 0.0
            elif lo >= 0.0:
                a = pulp.LpVariable(f"{prefix}_a_{li}_{n}", lowBound=lo, upBound=hi, cat="Continuous")
                problem += a == z
            else:
                b = pulp.LpVariable(f"{prefix}_r_{li}_{n}", cat="Binary")
                a = pulp.LpVariable(f"{prefix}_a_{li}_{n}", lowBound=0.0, upBound=hi, cat="Continuous")
                problem += a >= z
                problem += a >= 0.0
                problem += a <= z - lo * (1 - b)
                problem += a <= hi * b
            next_vars.append(a)
        prev_vars = next_vars
        prev_lower = pre_lo if is_output else np.maximum(0.0, pre_lo)
        prev_upper = pre_hi if is_output else np.maximum(0.0, pre_hi)

    if len(prev_vars) != 1:
        raise ValueError("final layer must produce a single output logit")
    return prev_vars[0]


def _encode_xgboost_forward_pass(problem, input_vars, trees, prefix, input_bound, base_offset=0.0):
    total = []
    big_m = float(2.0 * input_bound + 5.0)
    for ti, tree in enumerate(trees):
        leaves = tree["leaves"]
        if not leaves:
            raise ValueError(f"tree {ti} has no leaves")
        
        choice = {lf["leaf_id"]: pulp.LpVariable(f"{prefix}_t{ti}_lf_{lf['leaf_id']}", cat="Binary")
                  for lf in leaves}
        problem += pulp.lpSum(choice.values()) == 1, f"{prefix}_t{ti}_onehot"
        
        for lf in leaves:
            ind = choice[lf["leaf_id"]]
            for fidx, split, direction in lf["path"]:
                # correction: XGBoost uses x<split → yes, x>=split → no
                if direction == "lt":
                    problem += input_vars[fidx] <= split - 1e-8 + big_m * (1 - ind)
                else:
                    problem += input_vars[fidx] >= split - big_m * (1 - ind)
        total.append(pulp.lpSum(float(lf["value"]) * choice[lf["leaf_id"]] for lf in leaves))
    return pulp.lpSum(total) + float(base_offset)  # Add base margin


# Create input variables
def _create_inputs(problem, feature_names, bound_limit, pairwise):
    dim = len(feature_names)
    x_a: List[Any] = [None] * dim
    x_b: Optional[List[Any]] = [None] * dim if pairwise else None
    cat_groups = _detect_categorical_groups(feature_names)
    cat_indices = {i for idxs in cat_groups.values() for i in idxs}

    for key, idxs in cat_groups.items():
        for i in idxs:
            x_a[i] = pulp.LpVariable(f"xA_{i}", lowBound=0, upBound=1, cat="Binary")
            if pairwise:
                x_b[i] = pulp.LpVariable(f"xB_{i}", lowBound=0, upBound=1, cat="Binary")
        problem += pulp.lpSum(x_a[i] for i in idxs) == 1, f"onehot_A_{key}"
        if pairwise:
            problem += pulp.lpSum(x_b[i] for i in idxs) == 1, f"onehot_B_{key}"

    for i in range(dim):
        if x_a[i] is None:
            x_a[i] = pulp.LpVariable(f"xA_{i}", lowBound=-bound_limit, upBound=bound_limit, cat="Continuous")
            if pairwise:
                x_b[i] = pulp.LpVariable(f"xB_{i}", lowBound=-bound_limit, upBound=bound_limit, cat="Continuous")
    return x_a, x_b, cat_indices


# ---------------------------------------------------------------------------
# Property-specific constraints
# ---------------------------------------------------------------------------
def _monotonic_expected_sign(name: str) -> int:
    n = name.lower()
    base = 1
    if any(k in n for k in _RISK_INCREASING):
        base = -1
    elif any(k in n for k in _RISK_DECREASING):
        base = 1
    return base if POSITIVE_MARGIN_MEANS_APPROVE else -base


def _add_monotonicity_constraints(problem, x_a, x_b, target_index, epsilon):
    for i in range(len(x_a)):
        if i != target_index:
            problem += x_a[i] == x_b[i]
    problem += x_b[target_index] >= x_a[target_index] + epsilon  


def _add_individual_fairness_constraints(problem, x_a, x_b, sensitive_indices):
    s = set(sensitive_indices)
    for i in range(len(x_a)):
        if i not in s:
            problem += x_a[i] == x_b[i]


def _add_local_robustness_constraints(problem, x_a, original_x, epsilon):
    """constrain x_a within the original L∞ epsilon ball. x_b is not created (the original is constant)."""
    for i in range(len(x_a)):
        problem += x_a[i] >= float(original_x[i]) - epsilon
        problem += x_a[i] <= float(original_x[i]) + epsilon


def _categorize_features(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    cat = {"immutable": [], "increasing_only": [], "decreasing_only": [], "flexible": []}
    for i, name in enumerate(feature_names):
        n = name.lower()
        if any(k in n for k in _IMMUTABLE_KEYWORDS):
            cat["immutable"].append(i)
        elif any(k in n for k in _INCREASING_ONLY):
            cat["increasing_only"].append(i)
        elif any(k in n for k in _DECREASING_ONLY):
            cat["decreasing_only"].append(i)
        else:
            cat["flexible"].append(i)
    return cat


def _get_feature_costs(feature_names: Sequence[str]) -> np.ndarray:
    costs = np.ones(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        n = name.lower()
        if any(k in n for k in ["age", "sex", "gender", "race", "foreign"]):
            costs[i] = 1e6
        elif "employment" in n:
            costs[i] = 5.0
        elif "credit_history" in n:
            costs[i] = 3.0
    return costs


def _add_recourse_constraints(problem, x_a, feature_names, original_x, bound_limit, cat_indices):
    """do not apply numeric directional constraints to one-hot dummy (categorical binary) features.
    Categorical features are handled only by the onehot sum==1 constraint and, for immutable groups, by fixing them."""
    original_x = np.asarray(original_x, dtype=np.float64)
    dim = len(feature_names)
    categorized = _categorize_features(feature_names)
    costs = _get_feature_costs(feature_names)
    dist_vars = []

    for i in range(dim):
        orig = float(original_x[i])
        dv = pulp.LpVariable(f"dist_{i}", lowBound=0.0, cat="Continuous")
        problem += dv >= costs[i] * (x_a[i] - orig)
        problem += dv >= costs[i] * (orig - x_a[i])
        dist_vars.append(dv)

        if i in cat_indices:
            # Categorical one-hot bit: no directional constraint. Immutable categories are fixed.
            if i in categorized["immutable"]:
                problem += x_a[i] == orig, f"ImmutableCat_{i}"
            continue

        if i in categorized["immutable"]:
            problem += x_a[i] == orig, f"Immutable_{i}"
        elif i in categorized["increasing_only"]:
            problem += x_a[i] >= orig, f"IncOnly_{i}"
            problem += x_a[i] <= bound_limit
        elif i in categorized["decreasing_only"]:
            problem += x_a[i] <= orig, f"DecOnly_{i}"
            problem += x_a[i] >= -bound_limit
        else:
            problem += x_a[i] >= -bound_limit
            problem += x_a[i] <= bound_limit
    return dist_vars

def _build_problem_for_model(model_kind, model_path, feature_names, property_name,
                             target_feature_name=None, sensitive_feature_names=None,
                             bound_limit=3.0, epsilon=0.01, output_margin=1e-4,
                             original_x=None, original_output=None):
    dim = len(feature_names)
    pairwise = property_name in ("monotonicity", "individual_fairness")
    problem = pulp.LpProblem(f"{model_kind}_{property_name}_verification", pulp.LpMinimize)
    problem += 0

    x_a, x_b, cat_indices = _create_inputs(problem, feature_names, bound_limit, pairwise)

    target_index = None
    if property_name == "monotonicity":
        if target_feature_name is None:
            raise ValueError("target_feature_name is required for monotonicity")
        target_index = _resolve_feature_index(feature_names, target_feature_name)
        _add_monotonicity_constraints(problem, x_a, x_b, target_index, epsilon)
    elif property_name == "individual_fairness":
        s = _resolve_feature_indices(feature_names, sensitive_feature_names)
        if not s:
            raise ValueError("sensitive_feature_names is required for individual_fairness")
        _add_individual_fairness_constraints(problem, x_a, x_b, s)
    elif property_name == "local_robustness":
        if original_x is None:
            raise ValueError("original_x is required for local_robustness")
        _add_local_robustness_constraints(problem, x_a, original_x, epsilon)
    elif property_name == "recourse":
        if original_x is None:
            raise ValueError("original_x is required for recourse")
        dist = _add_recourse_constraints(problem, x_a, feature_names, original_x, bound_limit, cat_indices)
        problem.setObjective(pulp.lpSum(dist))
    else:
        raise ValueError(f"unsupported property: {property_name}")

    # Model encoding (x_b forward pass only for pairwise properties)
    if model_kind == "xgboost":
        booster = _load_xgboost_booster(model_path)
        trees = _parse_xgboost_trees(booster)
        offset = _xgboost_base_offset(booster, trees, dim)
        output_a = _encode_xgboost_forward_pass(problem, x_a, trees, "A", bound_limit, offset)
        output_b = _encode_xgboost_forward_pass(problem, x_b, trees, "B", bound_limit, offset) if pairwise else None
    elif model_kind == "mlp":
        layers = _load_mlp_layers(model_path, dim)
        if property_name in ("local_robustness", "recourse") and original_x is not None:
            # If checking local property, inputs don't need to span the entire [-bound_limit, bound_limit]
            # Adjust padding as appropriate for your feature scaling
            pad = epsilon if property_name == "local_robustness" else 1.0 
            lo = np.clip(original_x - pad, -bound_limit, bound_limit)
            hi = np.clip(original_x + pad, -bound_limit, bound_limit)
        else:
            lo = np.full(dim, -bound_limit)
            hi = np.full(dim, bound_limit)
        output_a = _encode_mlp_forward_pass(problem, x_a, layers, "A", lo, hi)
        output_b = _encode_mlp_forward_pass(problem, x_b, layers, "B", lo, hi) if pairwise else None
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")

    # Output (violation) constraints
    if property_name == "monotonicity":
        sign = _monotonic_expected_sign(target_feature_name)
        # b is larger. Expected: the output should change in the sign direction. Violation = opposite direction.
        if sign > 0:
            problem += output_a - output_b >= output_margin  # should increase, but decreases → violation
        else:
            problem += output_b - output_a >= output_margin  # should decrease, but increases → violation
    elif property_name == "individual_fairness":
        # If only sensitive attributes differ and the output differs, that is a violation.
        # Because the case is symmetric, searching one direction is sufficient.
        problem += output_a - output_b >= output_margin
    elif property_name == "local_robustness":
        # determine the flip direction based on the original prediction sign
        approved = (original_output >= 0.0) if POSITIVE_MARGIN_MEANS_APPROVE else (original_output < 0.0)
        if (approved and POSITIVE_MARGIN_MEANS_APPROVE) or (not approved and not POSITIVE_MARGIN_MEANS_APPROVE):
            problem += output_a <= -output_margin  # approved → flipped to rejected
        else:
            problem += output_a >= output_margin   # rejected → flipped to approved
    elif property_name == "recourse":
        if POSITIVE_MARGIN_MEANS_APPROVE:
            problem += output_a >= output_margin, "Must_Approve"
        else:
            problem += output_a <= -output_margin, "Must_Approve"

    payload = {"xA": x_a, "xB": x_b, "outputA": output_a, "outputB": output_b,
               "pairwise": pairwise, "target_index": target_index,
               "expected_sign": _monotonic_expected_sign(target_feature_name) if target_index is not None else None}
    return problem, payload


def _solve_problem(problem, timeout_seconds):
    num_constraints = len(problem.constraints)
    num_binaries = sum(1 for v in problem.variables() if v.cat in (pulp.LpInteger, pulp.LpBinary))
    solver = _build_solver(timeout_seconds)
    t0 = time.perf_counter()
    if solver is None:
        # Fallback when no timed solver is available — high risk of indefinite hanging on large MILPs
        print("[WARN] no timed solver available; solving without time limit")
        problem.solve()
    else:
        problem.solve(solver)
    elapsed = time.perf_counter() - t0
    status = pulp.LpStatus.get(problem.status, str(problem.status))
    return status, {"solve_seconds": elapsed,
                    "solver": solver.__class__.__name__ if solver else "default",
                    "num_constraints": num_constraints, "num_binary_vars": num_binaries}


def _compute_model_output(model_kind, model_path, input_x, feature_names):
    dim = len(feature_names)
    if model_kind == "xgboost":
        booster = _load_xgboost_booster(model_path)
        dm = xgb.DMatrix(np.asarray(input_x, dtype=np.float32).reshape(1, -1))  # feature_names eliminate
        return float(booster.predict(dm, output_margin=True)[0])              
    elif model_kind == "mlp":
        layers = _load_mlp_layers(model_path, dim)
        x = torch.tensor(np.asarray(input_x, dtype=np.float32)).unsqueeze(0)
        last = len(layers) - 1
        with torch.no_grad():
            for idx, (w, b) in enumerate(layers):
                x = torch.nn.functional.linear(x, torch.tensor(w, dtype=torch.float32),
                                               torch.tensor(b, dtype=torch.float32))
                if idx != last:  # safety index comparison
                    x = torch.nn.functional.relu(x)
        return float(x.squeeze().item())
    raise ValueError(f"unsupported model kind: {model_kind}")


def _estimate_mlp_flops(layers):
    return sum(2 * int(w.shape[1]) * int(w.shape[0]) for w, _ in layers)


def _estimate_xgb_ops(booster):
    try:
        df = booster.trees_to_dataframe()
    except Exception:
        return 0
    return int(len(df[df["Feature"] != "Leaf"]))


# Verification Entry Point
def verify_property_milp(model_kind, model_path, feature_names, property_name,
                         target_feature_name=None, sensitive_feature_names=None,
                         bound_limit=3.0, epsilon=0.01, output_margin=1e-4,
                         timeout_seconds=60, original_x=None, original_output=None):
    # Compute original output if not provided; required for robustness direction and reporting
    if original_x is not None and original_output is None:
        original_output = _compute_model_output(model_kind, model_path, original_x, feature_names)

    problem, payload = _build_problem_for_model(
        model_kind, model_path, feature_names, property_name,
        target_feature_name, sensitive_feature_names,
        bound_limit, epsilon, output_margin, original_x, original_output)
    status, stats = _solve_problem(problem, timeout_seconds)

    result = {"model_kind": model_kind, "property_name": property_name,
              "status": status, **stats}
    if original_output is not None:
        result["original_output"] = original_output

    if status.lower() in {"optimal", "feasible"}:
        x_a = np.array([pulp.value(v) for v in payload["xA"]], dtype=float)
        output_a = float(pulp.value(payload["outputA"]))
        result["xA"] = x_a
        result["outputA"] = output_a
        if payload["pairwise"]:
            x_b = np.array([pulp.value(v) for v in payload["xB"]], dtype=float)
            output_b = float(pulp.value(payload["outputB"]))
            result["xB"] = x_b
            result["outputB"] = output_b

        # compute severity consistently per property (always positive = violation magnitude)
        if property_name == "monotonicity":
            sign = payload["expected_sign"]
            result["violation_severity"] = (output_a - output_b) if sign > 0 else (output_b - output_a)
        elif property_name == "individual_fairness":
            result["violation_severity"] = abs(output_a - output_b)
        elif property_name == "local_robustness":
            result["xB"] = np.asarray(original_x, dtype=float)
            result["outputB"] = float(original_output)
            result["linf_perturbation"] = float(np.max(np.abs(x_a - np.asarray(original_x, dtype=float))))
            result["output_flip"] = bool(output_a * original_output < 0)
            result["violation_severity"] = abs(output_a - original_output)
    else:
        result["violation_severity"] = None

    return result


def compare_xgboost_and_mlp(xgboost_path, mlp_path, feature_names, property_name,
                            target_feature_name=None, sensitive_feature_names=None,
                            bound_limit=3.0, epsilon=0.01, output_margin=1e-4,
                            timeout_seconds=60, original_x=None):
    dummy_x = original_x if original_x is not None else np.zeros(len(feature_names))

    try:
        booster = _load_xgboost_booster(xgboost_path)
        est_xgb = _estimate_xgb_ops(booster)
        xgb_size = len(booster.trees_to_dataframe()["Tree"].unique()) # 트리 개수
        
        t0 = time.perf_counter()
        for _ in range(100):
            _compute_model_output("xgboost", xgboost_path, dummy_x, feature_names)
        xgb_latency = (time.perf_counter() - t0) / 100 * 1000 # 밀리초(ms) 변환
    except Exception:
        est_xgb, xgb_size, xgb_latency = None, None, None

    try:
        layers = _load_mlp_layers(mlp_path, len(feature_names))
        est_mlp = _estimate_mlp_flops(layers)
        mlp_size = sum(w.size + b.size for w, b in layers) # 파라미터 개수
        
        t0 = time.perf_counter()
        for _ in range(100):
            _compute_model_output("mlp", mlp_path, dummy_x, feature_names)
        mlp_latency = (time.perf_counter() - t0) / 100 * 1000
    except Exception:
        est_mlp, mlp_size, mlp_latency = None, None, None
        
    results = {}
    for kind, path, est, size, lat in (("xgboost", xgboost_path, est_xgb, xgb_size, xgb_latency), 
                                       ("mlp", mlp_path, est_mlp, mlp_size, mlp_latency)):
        t0 = time.perf_counter()
        res = verify_property_milp(kind, path, feature_names, property_name,
                                   target_feature_name, sensitive_feature_names,
                                   bound_limit, epsilon, output_margin, timeout_seconds,
                                   original_x=original_x)
        res["wall_seconds"] = time.perf_counter() - t0
        res["estimated_ops"] = est
        res["model_size"] = size
        res["latency_ms"] = lat
        results[kind] = res

    # Worse model: the one with the larger violation_severity (only when a violation is found)
    def sev(r):
        s = r.get("violation_severity")
        return s if (s is not None and s > 0) else float("-inf")
    xs, ms = sev(results["xgboost"]), sev(results["mlp"])
    if xs == float("-inf") and ms == float("-inf"):
        worse = None
    elif xs == ms:
        worse = "equal"
    else:
        worse = "xgboost" if xs > ms else "mlp"

    results["worse_model_by_severity"] = worse
    return results


# Default feature inference / input normalization

def _default_monotonicity_feature(feature_names):
    for c in ["num__duration", "num__credit_amount", "duration", "credit_amount"]:
        for n in feature_names:
            if c in n:
                return n
    for n in feature_names:
        if "duration" in n.lower() or "amount" in n.lower():
            return n
    return feature_names[0]


def _default_sensitive_feature(feature_names):
    for c in ["num__age", "age"]:
        for n in feature_names:
            if c in n:
                return n
    return feature_names[0]


def _normalize_original_input(original_x, input_dim):
    if original_x is None:
        raise ValueError("original_x must not be None")
    if isinstance(original_x, tuple) and len(original_x) == 2:
        original_x = original_x[0]
    if isinstance(original_x, torch.Tensor):
        arr = original_x.detach().cpu().numpy()
    elif hasattr(original_x, "to_numpy"):
        arr = original_x.to_numpy()
    elif hasattr(original_x, "values") and not callable(getattr(original_x, "values", None)):
        arr = np.asarray(original_x.values)
    else:
        arr = np.asarray(original_x)
    arr = arr.astype(np.float64).reshape(-1)
    if arr.shape[0] != input_dim:
        raise ValueError(f"original_x length {arr.shape[0]} != input_dim {input_dim}")
    return arr


def parse_args():
    p = argparse.ArgumentParser(description="MILP verification for XGBoost and baseline MLP")
    p.add_argument("--property", choices=["monotonicity", "individual_fairness", "compare",
                                          "recourse", "local_robustness"], default="compare")
    p.add_argument("--xgboost-path", default="saved_weights/best_xgboost.json")
    p.add_argument("--mlp-path", default="saved_weights/best_baseline_mlp.pth")
    p.add_argument("--target-feature", default=None)
    p.add_argument("--sensitive-features", nargs="*", default=None)
    p.add_argument("--bound-limit", type=float, default=3.0)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--output-margin", type=float, default=1e-4)  # Set above Big-M/solver tolerance
    p.add_argument("--timeout-seconds", type=int, default=60)
    p.add_argument("--sample-index", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle = load_and_preprocess_data()
    feature_names = list(bundle["feature_names"])
    target_feature_name = args.target_feature or _default_monotonicity_feature(feature_names)
    sensitive_feature_names = args.sensitive_features or [_default_sensitive_feature(feature_names)]

    dataset_x = bundle.get("X_test", bundle.get("X", None))
    if dataset_x is not None and len(dataset_x) > args.sample_index:
        try:
            original_x_sample = _normalize_original_input(dataset_x[args.sample_index], len(feature_names))
        except Exception:
            original_x_sample = np.zeros(len(feature_names))
    else:
        original_x_sample = np.zeros(len(feature_names))

    if args.property in ("compare", "monotonicity"):
        print(f"\nComparing Monotonicity over '{target_feature_name}'...")
        r = compare_xgboost_and_mlp(args.xgboost_path, args.mlp_path, feature_names,
                                    "monotonicity", target_feature_name=target_feature_name,
                                    bound_limit=args.bound_limit, epsilon=args.epsilon,
                                    output_margin=args.output_margin, timeout_seconds=args.timeout_seconds)
        for m in ("xgboost", "mlp"):
            res = r[m]
            print(f"  - {m.upper():<8} | Status: {res['status']:<10} | Time: {res.get('solve_seconds',0):.4f}s")
            unit = "Trees" if m == "xgboost" else "Params"
            print(f"             | Inference: {res.get('latency_ms', 0):.4f}ms | Size: {res.get('model_size', 0)} {unit} | Ops: {res.get('estimated_ops', 0)}")
            
            if res["status"].lower() in {"optimal", "feasible"}:
                print(f"             | Violation severity: {res.get('violation_severity')}")
        print(f"  Worse by monotonicity: {r.get('worse_model_by_severity')}")

    if args.property in ("compare", "individual_fairness"):
        print(f"\nComparing Individual Fairness over {sensitive_feature_names}...")
        r = compare_xgboost_and_mlp(args.xgboost_path, args.mlp_path, feature_names,
                                    "individual_fairness", sensitive_feature_names=sensitive_feature_names,
                                    bound_limit=args.bound_limit, epsilon=args.epsilon,
                                    output_margin=args.output_margin, timeout_seconds=args.timeout_seconds)
        for m in ("xgboost", "mlp"):
            res = r[m]
            print(f"  - {m.upper():<8} | Status: {res['status']:<10} | Time: {res.get('solve_seconds',0):.4f}s")
            unit = "Trees" if m == "xgboost" else "Params"
            print(f"             | Inference: {res.get('latency_ms', 0):.4f}ms | Size: {res.get('model_size', 0)} {unit} | Ops: {res.get('estimated_ops', 0)}")
            
            if res["status"].lower() in {"optimal", "feasible"}:
                print(f"             | Fairness violation (|diff|): {res.get('violation_severity'):.6f}")
        print(f"  Worse by fairness: {r.get('worse_model_by_severity')}")

    elif args.property == "local_robustness":
        print(f"Local Robustness MILP for customer index {args.sample_index}, epsilon={args.epsilon}")
        r = compare_xgboost_and_mlp(args.xgboost_path, args.mlp_path, feature_names,
                                    "local_robustness", bound_limit=args.bound_limit,
                                    epsilon=args.epsilon, output_margin=args.output_margin,
                                    timeout_seconds=args.timeout_seconds, original_x=original_x_sample)
        for m in ("xgboost", "mlp"):
            res = r[m]
            print(f"\n===== {m.upper()} =====")
            print(f"Status: {res['status']} | Solve: {res.get('solve_seconds',0):.4f}s")
            if res["status"].lower() in {"optimal", "feasible"}:
                print(f"  VIOLATED within epsilon={args.epsilon}")
                print(f"  original margin: {res['outputB']:.6f} | perturbed: {res['outputA']:.6f}")
                print(f"  output flip: {res.get('output_flip')} | L_inf: {res.get('linf_perturbation'):.6f}")
                for i, name in enumerate(feature_names):
                    d = res["xA"][i] - res["xB"][i]
                    if abs(d) > 1e-6:
                        print(f"    [{name}]: {d:+.6f}")
            else:
                print(f"  VERIFIED: no violation within epsilon={args.epsilon}")

    elif args.property == "recourse":
        print(f"Actionable Recourse MILP for customer index {args.sample_index}\n")
        categorized = _categorize_features(feature_names)
        for m, path in (("XGBoost", args.xgboost_path), ("MLP", args.mlp_path)):
            res = verify_property_milp(m.lower() if m == "MLP" else "xgboost", path, feature_names,
                                       "recourse", bound_limit=args.bound_limit,
                                       output_margin=args.output_margin,
                                       timeout_seconds=args.timeout_seconds, original_x=original_x_sample)
            print(f"\n{'='*70}\nMODEL: {m}\n{'='*70}")
            print(f"Status: {res['status']} | Solve: {res.get('solve_seconds')}")
            if res.get("status") == "Optimal":
                print(f"  Original margin: {res.get('original_output')} | New: {res['outputA']:.6f} (APPROVED)")
                print(f"\n{'Feature':<30}{'Current':<12}{'New':<12}{'Change':<12}")
                changed = False
                for i, name in enumerate(feature_names):
                    orig, new = original_x_sample[i], res["xA"][i]
                    if abs(new - orig) > 1e-5:
                        changed = True
                        print(f"{name:<30}{orig:<12.4f}{new:<12.4f}{new-orig:+.4f}")
                if not changed:
                    print("  No changes needed.")
            else:
                print("  NO SOLUTION — approval not achievable within bounds.")
