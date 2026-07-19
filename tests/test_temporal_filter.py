import pytest

from src.temporal_filter import TemporalRiskFilter


def test_temporal_risk_filter_can_be_created() -> None:
    risk_filter = TemporalRiskFilter(window_size=5, min_risk_count=3)

    state = risk_filter.get_state()

    assert state["window_size"] == 5
    assert state["min_risk_count"] == 3


def test_temporal_risk_filter_rejects_zero_window_size() -> None:
    with pytest.raises(ValueError):
        TemporalRiskFilter(window_size=0, min_risk_count=1)


def test_temporal_risk_filter_rejects_zero_min_risk_count() -> None:
    with pytest.raises(ValueError):
        TemporalRiskFilter(window_size=5, min_risk_count=0)


def test_temporal_risk_filter_rejects_min_risk_count_greater_than_window_size() -> None:
    with pytest.raises(ValueError):
        TemporalRiskFilter(window_size=3, min_risk_count=4)


def test_temporal_risk_filter_returns_stable_true_when_threshold_is_met() -> None:
    risk_filter = TemporalRiskFilter(window_size=5, min_risk_count=3)
    result = None

    for raw_risk in [True, False, True, True]:
        result = risk_filter.update(raw_risk)

    assert result["stable_risk"] is True
    assert result["risk_count"] == 3


def test_temporal_risk_filter_returns_stable_false_when_threshold_is_not_met() -> None:
    risk_filter = TemporalRiskFilter(window_size=5, min_risk_count=3)
    result = None

    for raw_risk in [True, False, False, False, True]:
        result = risk_filter.update(raw_risk)

    assert result["stable_risk"] is False
    assert result["risk_count"] == 2


def test_temporal_risk_filter_reset_clears_history() -> None:
    risk_filter = TemporalRiskFilter(window_size=5, min_risk_count=3)
    risk_filter.update(True)

    risk_filter.reset()

    assert risk_filter.get_state()["history"] == []


def test_temporal_risk_filter_get_state_returns_current_values() -> None:
    risk_filter = TemporalRiskFilter(window_size=5, min_risk_count=3)
    risk_filter.update(True)
    risk_filter.update(False)

    state = risk_filter.get_state()

    assert state["history"] == [True, False]
    assert state["risk_count"] == 1
    assert state["window_size"] == 5
    assert state["min_risk_count"] == 3
