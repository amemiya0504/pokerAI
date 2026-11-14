"""End-to-end: play full hands in a real browser on desktop and mobile.

Needs the built frontend (``npm run build`` in frontend/) and Playwright's
Chromium (``python -m playwright install chromium``). Skipped otherwise.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from backend.api.app import FRONTEND_DIST, create_app
from backend.engine.cards import card_str

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e

VIEWPORTS = {
    "desktop": {"viewport": {"width": 1280, "height": 800}},
    "mobile": {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True,
               "device_scale_factor": 2},
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not (FRONTEND_DIST / "index.html").exists():
        pytest.skip("frontend is not built (run npm run build in frontend/)")
    import uvicorn

    db = tmp_path_factory.mktemp("e2e") / "e2e.db"
    app = create_app(f"sqlite:///{db}", ai_delay=0.05)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield app, f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # browser or its system libraries missing
            pytest.skip(f"cannot launch chromium: {e}")
        yield b
        b.close()


def frame_cards(payload: str) -> tuple[set[str], bool]:
    """Card strings in one websocket frame, and whether the hand was over."""
    msg = json.loads(payload)
    hand = msg.get("hand") or {}
    found: set[str] = set(hand.get("hole_cards", [])) | set(hand.get("board", []))
    for st in hand.get("seats", {}).values():
        found |= set(st.get("shown_cards") or [])
    for e in hand.get("events", []):
        found |= set(e.get("cards", []))
    return found, bool(hand.get("is_over"))


@pytest.mark.parametrize("device", list(VIEWPORTS))
def test_play_full_hands(server, browser, device):
    app, url = server
    ctx = browser.new_context(**VIEWPORTS[device])
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    frames: list[str] = []
    page.on("websocket", lambda ws: ws.on("framereceived", lambda p: frames.append(p)))

    page.goto(url)
    page.select_option("[data-testid=players]", "4")
    page.click("[data-testid=create-submit]")
    page.wait_for_url("**/#/game/*")
    game_id = int(page.url.rsplit("/", 1)[1])
    session = app.state.get_session(game_id)

    for hand_no in (1, 2):
        frames.clear()
        page.click("[data-testid=next-hand]")
        # Wait for the new hand to replace the previous hand's result.
        page.wait_for_selector("[data-testid=result]", state="detached")
        page.wait_for_function(
            "n => document.querySelector('[data-testid=status]').textContent.includes(`第 ${n} 手`)",
            arg=hand_no,
        )
        dom_leaks: list[set[str]] = []
        deadline = time.time() + 60
        while time.time() < deadline:
            if page.locator("[data-testid=result]").count() and page.locator("[data-testid=next-hand], [data-testid=rebuy]").count():
                break
            for testid in ("action-check", "action-call"):
                btn = page.locator(f"[data-testid={testid}]")
                if btn.count():
                    # Snapshot every face-up card in the DOM at our turn.
                    dom_leaks.append(set(page.eval_on_selector_all(
                        "[data-card]", "els => els.map(e => e.dataset.card)")) - {"hidden"})
                    btn.click()
                    break
            page.wait_for_timeout(50)
        else:
            pytest.fail("hand did not finish")

        state = session.table.last_hand
        assert state is not None and session.table.hand_no == hand_no
        hero = session.human_seat
        hero_cards = {card_str(c) for c in state.hole_cards(hero)}
        board = {card_str(c) for c in state.board}
        for seat in state.seats:
            if seat == hero:
                continue
            hidden = {card_str(c) for c in state.hole_cards(seat)}
            terminal_cards: set[str] = set()
            for payload in frames:
                cards, over = frame_cards(payload)
                if over:
                    terminal_cards |= cards
                else:
                    assert not hidden & cards, f"seat {seat} cards reached the browser before the hand ended"
            assert hidden <= terminal_cards, f"seat {seat} cards were not revealed after the hand ended"
            for seen in dom_leaks:
                assert not hidden & seen, f"seat {seat}'s cards rendered during play"
        for seen in dom_leaks:
            assert seen <= hero_cards | board

        assert page.evaluate("document.documentElement.scrollWidth") <= page.viewport_size["width"]
        if page.locator("[data-testid=next-hand]").count() == 0:
            page.click("[data-testid=rebuy]")
            page.wait_for_selector("[data-testid=next-hand]")

    # History list and step-through replay.
    page.goto(f"{url}/#/game/{game_id}/history")
    page.wait_for_selector("[data-testid=hand-list] a")
    assert page.locator("[data-testid=hand-list] a").count() == 2
    page.locator("[data-testid=hand-list] a").last.click()
    page.wait_for_selector("[data-testid=replay-step]")
    page.click("[data-testid=replay-prev]")
    assert "第" in page.inner_text("[data-testid=replay-step]")
    assert not errors, errors
    ctx.close()
