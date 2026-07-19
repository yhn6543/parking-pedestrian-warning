import threading


class RealtimeAlertSound:
    def __init__(
        self,
        enabled: bool = True,
        trigger_mode: str = "stable",
        cooldown_frames: int = 10,
        frequency: int = 1000,
        duration_ms: int = 250,
    ):
        normalized_mode = str(trigger_mode).strip().lower()
        if normalized_mode not in {"logged", "stable_start", "stable"}:
            raise ValueError("trigger_mode must be 'logged', 'stable_start', or 'stable'.")

        self.enabled = bool(enabled)
        self.trigger_mode = normalized_mode
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.frequency = int(frequency)
        self.duration_ms = int(duration_ms)
        self.previous_stable_risk = False
        self.last_played_frame_index: int | None = None
        self.sound_count = 0

    def _cooldown_ready(self, frame_index: int) -> bool:
        if self.last_played_frame_index is None:
            return True
        return (int(frame_index) - self.last_played_frame_index) >= self.cooldown_frames

    def should_play(self, frame_index: int, stable_risk: bool, logged: bool) -> bool:
        stable_risk = bool(stable_risk)
        logged = bool(logged)

        if not self.enabled:
            self.previous_stable_risk = stable_risk
            return False

        if self.trigger_mode == "logged":
            triggered = logged
        elif self.trigger_mode == "stable_start":
            triggered = stable_risk and not self.previous_stable_risk
        else:
            triggered = stable_risk

        self.previous_stable_risk = stable_risk
        return bool(triggered and self._cooldown_ready(frame_index))

    def play(self) -> None:
        def play_beep() -> None:
            try:
                import winsound

                winsound.Beep(self.frequency, self.duration_ms)
            except Exception:
                try:
                    print("\a", end="", flush=True)
                except Exception:
                    pass

        thread = threading.Thread(target=play_beep, daemon=True)
        thread.start()

    def update_and_play(self, frame_index: int, stable_risk: bool, logged: bool) -> dict:
        sound_played = self.should_play(
            frame_index=frame_index,
            stable_risk=stable_risk,
            logged=logged,
        )
        if sound_played:
            self.last_played_frame_index = int(frame_index)
            self.sound_count += 1
            self.play()

        return {
            "sound_played": bool(sound_played),
            "trigger_mode": self.trigger_mode,
            "sound_count": int(self.sound_count),
        }
