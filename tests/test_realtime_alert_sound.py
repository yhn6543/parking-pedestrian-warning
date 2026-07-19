import pytest

from src.realtime_alert_sound import RealtimeAlertSound


def test_disabled_realtime_alert_sound_never_plays() -> None:
    sound = RealtimeAlertSound(enabled=False, trigger_mode="logged")

    assert sound.should_play(frame_index=1, stable_risk=True, logged=True) is False


def test_logged_trigger_only_plays_for_logged_alert() -> None:
    sound = RealtimeAlertSound(enabled=True, trigger_mode="logged")

    assert sound.should_play(frame_index=1, stable_risk=True, logged=False) is False
    assert sound.should_play(frame_index=2, stable_risk=False, logged=True) is True


def test_stable_start_trigger_plays_only_on_false_to_true_transition() -> None:
    sound = RealtimeAlertSound(enabled=True, trigger_mode="stable_start")

    assert sound.should_play(frame_index=1, stable_risk=False, logged=False) is False
    assert sound.should_play(frame_index=2, stable_risk=True, logged=False) is True
    assert sound.should_play(frame_index=3, stable_risk=True, logged=False) is False
    assert sound.should_play(frame_index=4, stable_risk=False, logged=False) is False
    assert sound.should_play(frame_index=5, stable_risk=True, logged=False) is True


def test_stable_trigger_respects_sound_cooldown() -> None:
    sound = RealtimeAlertSound(enabled=True, trigger_mode="stable", cooldown_frames=3)
    sound.play = lambda: None

    first = sound.update_and_play(frame_index=10, stable_risk=True, logged=False)
    second = sound.update_and_play(frame_index=12, stable_risk=True, logged=False)
    third = sound.update_and_play(frame_index=13, stable_risk=True, logged=False)

    assert first["sound_played"] is True
    assert second["sound_played"] is False
    assert third["sound_played"] is True
    assert sound.sound_count == 2


def test_update_and_play_calls_play_when_triggered(monkeypatch) -> None:
    calls = []
    sound = RealtimeAlertSound(enabled=True, trigger_mode="logged")
    monkeypatch.setattr(sound, "play", lambda: calls.append("play"))

    result = sound.update_and_play(frame_index=1, stable_risk=False, logged=True)

    assert result == {
        "sound_played": True,
        "trigger_mode": "logged",
        "sound_count": 1,
    }
    assert calls == ["play"]


def test_realtime_alert_sound_rejects_unknown_trigger_mode() -> None:
    with pytest.raises(ValueError, match="trigger_mode"):
        RealtimeAlertSound(trigger_mode="unknown")
