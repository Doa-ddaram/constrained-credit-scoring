import os

import xgboost as xgb
from sklearn.metrics import accuracy_score

from dataset import load_and_preprocess_data


def build_monotone_constraints(feature_names):
    """Create monotonic constraints for selected German Credit features."""
    # 0: no constraint, 1: monotonically increasing, -1: monotonically decreasing
    monotone_constraints = [0] * len(feature_names)

    for i, name in enumerate(feature_names):
        name_lower = name.lower()

        # Longer duration or larger credit amount tends to reduce approval likelihood.
        if "duration" in name_lower or "credit_amount" in name_lower:
            monotone_constraints[i] = -1
        # Higher savings/checking status tends to improve approval likelihood.
        elif "savings" in name_lower or "checking_status" in name_lower:
            monotone_constraints[i] = 1

    return monotone_constraints


def train_xgboost_with_monotone_constraints():
    data = load_and_preprocess_data()
    feature_names = data["feature_names"]
    save_dir = "saved_weights"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_xgboost.json")

    monotone_constraints = build_monotone_constraints(feature_names)
    print(
        f"Applied non-zero monotone constraints on "
        f"{sum(c != 0 for c in monotone_constraints)} / {len(feature_names)} features"
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        monotone_constraints=tuple(monotone_constraints),
        random_state=42,
    )

    print("Starting XGBoost training with monotonic constraints...")
    model.fit(data["X_train_np"], data["y_train_np"])

    y_pred = model.predict(data["X_test_np"])
    accuracy = accuracy_score(data["y_test_np"], y_pred)
    print(f"XGBoost training completed. Test accuracy: {accuracy * 100:.2f}%")

    model.save_model(best_model_path)
    print(f"Saved trained XGBoost model to '{best_model_path}'")

    return model, data


if __name__ == "__main__":
    train_xgboost_with_monotone_constraints()
