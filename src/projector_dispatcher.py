from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def dispatch_projector_alert(
    projector: dict | None,
    risk_event: dict | None = None,
    mode: str = "mock",
    timeout: float = 1.0,
) -> dict:
    if not projector:
        return {"status": "skipped", "reason": "no_projector"}

    mode = str(mode or "mock").lower()
    risk_event = risk_event or {}
    projector_id = str(projector.get("id", ""))

    if mode == "mock":
        return {
            "status": "mock_dispatched",
            "projector_id": projector_id,
            "projector_name": str(projector.get("name", "")),
        }

    if mode != "http":
        return {"status": "failed", "reason": f"unsupported_mode:{mode}", "projector_id": projector_id}

    endpoint = str(projector.get("endpoint", "") or "").strip()
    if not endpoint:
        return {"status": "skipped", "reason": "missing_endpoint", "projector_id": projector_id}

    payload: dict[str, Any] = {
        "projector": projector,
        "risk_event": risk_event,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return {
                "status": "http_dispatched",
                "projector_id": projector_id,
                "http_status": int(response.status),
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"status": "failed", "reason": str(exc), "projector_id": projector_id}
