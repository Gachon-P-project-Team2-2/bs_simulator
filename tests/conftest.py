"""pytest fixtures: Dash 서버 자동 시작/종료, Playwright 브라우저 컨텍스트."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STARTUP_TIMEOUT = 30  # 초


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.3)
    raise TimeoutError(f"Dash did not open {host}:{port} within {timeout}s")


@pytest.fixture(scope="session")
def dash_server():
    """세션 범위: 테스트 시작 시 Dash 앱을 백그라운드로 실행, 종료 시 정리."""
    port = _free_port()
    python = sys.executable
    env = os.environ.copy()
    env["DASH_PORT"] = str(port)

    proc = subprocess.Popen(
        [python, "app.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port("127.0.0.1", port, STARTUP_TIMEOUT)
        time.sleep(1)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def browser() -> Browser:
    """세션 범위 Playwright 브라우저 (Chromium)."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture()
def context(browser: Browser) -> BrowserContext:
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    yield ctx
    ctx.close()


@pytest.fixture()
def page(context: BrowserContext, dash_server: str) -> Page:
    """준비된 페이지: Dash URL로 이동 + 메인 UI 대기."""
    p = context.new_page()
    p.goto(dash_server)
    p.wait_for_selector("text=시뮬레이터 제어", timeout=40_000)
    p.wait_for_selector("text=가상 데이터 생성", timeout=20_000)
    return p
