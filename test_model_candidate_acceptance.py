import pytest

import s05_train_final_model as s05


def _candidate(name, accuracy, fp_rate, nodes=100, recall=0.9, cv_std=0.01,
               finite_predictions=True):
    return {
        "candidate": name,
        "valid_accuracy": accuracy,
        "valid_fp_rate": fp_rate,
        "valid_recall": recall,
        "total_nodes": nodes,
        "cv_accuracy_std": cv_std,
        "finite_predictions": finite_predictions,
    }


def test_model_candidate_selection_enforces_nodes_and_finite_predictions():
    records = [
        _candidate("too_large", 0.99, 0.0, nodes=501),
        _candidate("nonfinite", 0.99, 0.0, finite_predictions=False),
        _candidate("deployable", 0.95, 0.01, nodes=300),
    ]

    decision = s05.select_model_candidate(records, max_nodes=500, max_fpr=0.01)

    assert decision["selected_candidate"] == "deployable"
    rejected = {item["candidate"]: item["rejection_reasons"] for item in decision["leaderboard"]}
    assert "node_budget" in rejected["too_large"]
    assert "non_finite_predictions" in rejected["nonfinite"]


def test_model_candidate_selection_maximizes_accuracy_inside_fpr_constraint():
    records = [
        _candidate("lower_fp", 0.96, 0.002),
        _candidate("higher_accuracy", 0.98, 0.009),
        _candidate("infeasible", 0.995, 0.02),
    ]

    decision = s05.select_model_candidate(records, max_nodes=500, max_fpr=0.01)

    assert decision["selected_candidate"] == "higher_accuracy"
    assert decision["deployment_acceptance"] is True
    assert decision["selection_reason"] == "max_valid_accuracy_within_fpr_and_node_constraints"


def test_model_candidate_selection_tie_breaks_by_fpr_recall_nodes_and_stability():
    records = [
        _candidate("a", 0.98, 0.008, recall=0.91, nodes=100, cv_std=0.01),
        _candidate("b", 0.98, 0.006, recall=0.90, nodes=90, cv_std=0.005),
        _candidate("c", 0.98, 0.006, recall=0.92, nodes=120, cv_std=0.02),
        _candidate("d", 0.98, 0.006, recall=0.92, nodes=80, cv_std=0.01),
        _candidate("e", 0.98, 0.006, recall=0.92, nodes=80, cv_std=0.002),
    ]

    decision = s05.select_model_candidate(records)

    assert decision["selected_candidate"] == "e"


def test_model_candidate_selection_marks_analysis_only_when_fpr_target_unmet():
    records = [
        _candidate("accurate", 0.99, 0.04),
        _candidate("lowest_fp", 0.96, 0.02),
    ]

    decision = s05.select_model_candidate(records, max_fpr=0.01)

    assert decision["selected_candidate"] == "lowest_fp"
    assert decision["deployment_acceptance"] is False
    assert decision["status"] == "analysis_only"


@pytest.mark.parametrize(
    ("candidate_accuracy", "candidate_fpr", "accepted", "reason"),
    [
        (0.98, 0.009, True, "accuracy_not_lower_and_fpr_not_higher"),
        (0.979, 0.005, False, "valid_accuracy_decreased"),
        (0.98, 0.011, False, "valid_false_positive_rate_increased"),
    ],
)
def test_hard_negative_candidate_acceptance_requires_no_metric_regression(
        candidate_accuracy, candidate_fpr, accepted, reason):
    reference = _candidate("searched", 0.98, 0.01)
    candidate = _candidate("hard_negative", candidate_accuracy, candidate_fpr)

    decision = s05.accept_hard_negative_candidate(reference, candidate)

    assert decision["accepted"] is accepted
    assert decision["reason"] == reason
    assert decision["selected_candidate"] == (
        "hard_negative" if accepted else "searched"
    )
