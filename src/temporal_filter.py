from collections import deque


class TemporalRiskFilter:
    def __init__(self, window_size: int = 5, min_risk_count: int = 3):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if min_risk_count < 1:
            raise ValueError("min_risk_count must be at least 1.")
        if min_risk_count > window_size:
            raise ValueError("min_risk_count must be less than or equal to window_size.")

        self.window_size = int(window_size)
        self.min_risk_count = int(min_risk_count)
        self.history = deque(maxlen=self.window_size)

    def update(self, raw_risk: bool) -> dict:
        self.history.append(bool(raw_risk))
        return self.get_state(raw_risk=bool(raw_risk))

    def reset(self) -> None:
        self.history.clear()

    def get_state(self, raw_risk: bool | None = None) -> dict:
        history = list(self.history)
        risk_count = sum(1 for item in history if item)
        stable_risk = risk_count >= self.min_risk_count

        return {
            "raw_risk": bool(raw_risk) if raw_risk is not None else False,
            "stable_risk": bool(stable_risk),
            "risk_count": int(risk_count),
            "window_size": int(self.window_size),
            "min_risk_count": int(self.min_risk_count),
            "history": history,
        }
