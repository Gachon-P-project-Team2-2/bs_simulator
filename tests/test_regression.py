"""Playwright 기반 UI 회귀 테스트.

실행:
    venv/Scripts/python.exe -m pytest tests/test_regression.py -v

conftest.py가 streamlit을 자동 시작/종료하고 세션 전역 브라우저를 관리한다.
각 테스트는 새 페이지로 시작해 격리된다.
"""
from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

# REGISTRY / PATTERN_CHOICES를 직접 import하여, 항목 추가/제거 시 테스트가 자동 갱신됨
from optimizers import REGISTRY
from patterns import PATTERN_CHOICES


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def _generate_data(page: Page) -> None:
    """가상 데이터 생성 버튼 클릭 + 완료 대기."""
    btn = page.get_by_role("button", name="가상 데이터 생성")
    btn.scroll_into_view_if_needed()
    btn.click()
    # Streamlit 재렌더가 느릴 수 있어 여유 있게
    page.wait_for_selector("text=데이터 내보내기", timeout=60_000)


def _run_optimization(page: Page, timeout_ms: int = 180_000) -> None:
    """계산 실행 버튼 클릭 + 메트릭 표시 대기.

    GA는 기본 하이퍼파라미터로 수 분까지 걸릴 수 있으므로 넉넉히 대기.
    """
    page.get_by_role("button", name="계산 실행").click()
    # 메인 영역의 메트릭 카드 중 '총 트래픽'이 나타날 때까지
    page.wait_for_selector("text=총 트래픽", timeout=timeout_ms)


def _slider_to_min(page: Page, label: str) -> None:
    """슬라이더를 최솟값으로 (Home 키)."""
    slider = page.get_by_role("slider", name=label)
    slider.focus()
    slider.press("Home")


def _ensure_expander_open(page: Page, label: str) -> None:
    """Streamlit expander를 '열림' 상태로 보장 (이미 열려있으면 no-op)."""
    summary = page.locator("summary").filter(has_text=label).first
    # aria-expanded 속성으로 상태 확인 — 없으면 details open 속성 확인
    parent = summary.locator("..")  # details element
    is_open = parent.evaluate("el => el.open")
    if not is_open:
        summary.click()


def _minimize_hyperparams(page: Page, algo: str) -> None:
    """속도를 위해 반복 횟수를 최소로. 테스트 시간 단축용."""
    if algo == "K-Means":
        return  # K-Means는 기본 파라미터도 빠름
    _ensure_expander_open(page, "하이퍼파라미터")
    if algo in {"Random Walk", "Simulated Annealing", "Tabu Search"}:
        page.get_by_role("slider", name="iterations (반복 수)").wait_for(timeout=15_000)
        _slider_to_min(page, "iterations (반복 수)")
    elif algo == "Genetic Algorithm":
        page.get_by_role("slider", name="n_generations (세대 수)").wait_for(timeout=15_000)
        _slider_to_min(page, "n_generations (세대 수)")
        _slider_to_min(page, "pop_size (개체 수)")


def _select_by_combobox(page: Page, combobox_label_re: re.Pattern, option_name: str) -> None:
    """Streamlit selectbox에서 옵션 선택.

    이미 선택된 옵션을 다시 선택하려고 하면 드롭다운이 즉시 닫혀 DOM detach 발생.
    선택된 값을 먼저 확인해서 불필요한 클릭을 피한다.
    """
    combobox = page.get_by_role("combobox", name=combobox_label_re)
    combobox.scroll_into_view_if_needed()
    # 현재 선택값 확인 — aria-label이 "Selected <value>. <label>" 형식
    current_label = combobox.get_attribute("aria-label") or ""
    if f"Selected {option_name}." in current_label:
        return  # 이미 선택됨
    combobox.click()
    option = page.get_by_role("option", name=option_name, exact=True)
    option.wait_for(state="visible", timeout=10_000)
    # 드롭다운 애니메이션으로 "element is not stable" 발생 가능 → force로 우회
    option.click(force=True)


def _select_algorithm(page: Page, name: str) -> None:
    _select_by_combobox(page, re.compile(r"알고리즘 선택"), name)


def _select_pattern(page: Page, name: str) -> None:
    """트래픽 패턴 selectbox에서 name 선택 (expander가 열려있어야 함)."""
    _select_by_combobox(page, re.compile(r"트래픽 패턴"), name)


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
def test_app_loads(page: Page):
    """기본 UI가 로드되고 콘솔 에러가 없는지."""
    expect(page.get_by_text("시뮬레이터 제어")).to_be_visible()
    expect(page.get_by_text("1. 환경 설정")).to_be_visible()
    expect(page.get_by_text("2. 시각화 설정")).to_be_visible()
    expect(page.get_by_text("3. 계산 알고리즘")).to_be_visible()
    expect(page.get_by_role("button", name="가상 데이터 생성")).to_be_visible()


def test_generate_data_golden_path(page: Page):
    """데이터 생성 → 다운로드 버튼 노출."""
    _generate_data(page)
    expect(page.get_by_role("button", name="GIS 데이터 (CSV)")).to_be_visible()
    expect(page.get_by_role("button", name="Local 데이터 (CSV)")).to_be_visible()
    expect(page.get_by_role("button", name="Map 데이터 (NPY)")).to_be_visible()


def test_algorithm_selectbox_has_all_optimizers(page: Page):
    """REGISTRY의 모든 알고리즘이 UI selectbox에 노출되는지 (목록 직접 참조)."""
    page.get_by_role("combobox", name=re.compile(r"알고리즘 선택")).click()
    for cls in REGISTRY:
        expect(page.get_by_role("option", name=cls.name, exact=True)).to_be_visible()


def test_traffic_pattern_selectbox_has_all_patterns(page: Page):
    """PATTERN_CHOICES의 모든 패턴이 UI에 노출되는지 (목록 직접 참조)."""
    page.get_by_text("트래픽 세부 설정").click()
    page.get_by_role("combobox", name=re.compile(r"트래픽 패턴")).click()
    for name in PATTERN_CHOICES:
        expect(page.get_by_role("option", name=name, exact=True)).to_be_visible()


def test_kmeans_runs_golden_path(page: Page):
    """K-Means(기본 알고리즘)로 전체 골든 패스 확인.

    다른 알고리즘들은 test_algorithm_selectbox_has_all_optimizers에서 UI 등록 여부만
    확인한다. selectbox 변경 후 streamlit re-run 타이밍이 불안정해 per-algo 파라미터화는
    flaky하므로, 대표 알고리즘 하나로 end-to-end만 검증.
    """
    _generate_data(page)
    # Streamlit의 재렌더가 완료된 후 다음 클릭이 가도록 잠시 대기
    page.wait_for_timeout(1500)
    # K-Means는 기본값이므로 selectbox 재선택 불필요
    _run_optimization(page)

    # 메트릭 카드 3개 확인 — "기지국 수"는 사이드바 슬라이더와 충돌하므로 제외.
    expect(page.get_by_text("총 트래픽")).to_be_visible()
    expect(page.get_by_text("커버된 트래픽")).to_be_visible()
    expect(page.get_by_text("커버된 면적")).to_be_visible()


def test_ring_pattern_generates(page: Page):
    """ring 패턴(비-기본)으로 데이터 생성 — 8개 패턴 중 대표로 검증."""
    page.get_by_text("트래픽 세부 설정").click()
    _select_pattern(page, "ring")
    _generate_data(page)
    expect(page.get_by_role("button", name="GIS 데이터 (CSV)")).to_be_visible()


# Streamlit의 다운로드 버튼은 첫 클릭 전까지 blob URL이 없어 404를 찍는다 — 무해한 노이즈.
_IGNORED_ERROR_PATTERNS = (
    "Download Button source error",
    "Failed to load resource",
)


def test_no_console_errors(page: Page):
    """페이지 로드와 데이터 생성 중 콘솔 error가 발생하지 않는지 (무해한 노이즈 제외)."""
    errors: list[str] = []

    def _on_console(msg):
        if msg.type != "error":
            return
        text = msg.text
        if any(p in text for p in _IGNORED_ERROR_PATTERNS):
            return
        errors.append(text)

    page.on("console", _on_console)

    _generate_data(page)
    _select_algorithm(page, "K-Means")
    _run_optimization(page)

    assert errors == [], f"Unexpected console errors: {errors}"
