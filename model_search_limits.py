"""Shared limits for deployable XGBoost model-parameter search."""

MAX_MODEL_SEARCH_N_ESTIMATORS = 50
DEFAULT_MODEL_SEARCH_N_ESTIMATORS = (20, 25, 30, 35, 40, 45, 50)


def parse_model_search_n_estimators(raw):
    """Parse an ordered unique tree-count grid and enforce the hard cap."""
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model_search_n_estimators contains invalid value {part!r}"
            ) from exc
        if value > MAX_MODEL_SEARCH_N_ESTIMATORS:
            raise ValueError(
                "model_search_n_estimators maximum is "
                f"{MAX_MODEL_SEARCH_N_ESTIMATORS}; got {value}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("model_search_n_estimators must contain at least one value")
    return values


def default_model_search_n_estimators_csv():
    """Return the canonical CLI representation of the default tree grid."""
    return ",".join(str(value) for value in DEFAULT_MODEL_SEARCH_N_ESTIMATORS)
