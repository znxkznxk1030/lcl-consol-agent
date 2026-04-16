"""
olist_calibration.py
====================
Olist Brazilian E-Commerce 데이터셋(archive/)에서
시뮬레이터용 arrival_rates 와 CBM/무게 분포 파라미터를 추출한다.

사용법:
    from simulator_v1.olist_calibration import make_olist_config
    cfg = make_olist_config(archive_dir="../../archive", total_rate=6.0)

Olist 카테고리 → ItemType 매핑 근거:
  ELECTRONICS   : electronics, computers, tablets_printing_image,
                  consoles_games, audio, telephony, fixed_telephony,
                  computers_accessories, cine_photo
  CLOTHING      : fashion_*, luggage_accessories,
                  bed_bath_table (섬유·침구류), housewares (소형 생활용품)
  COSMETICS     : health_beauty, perfumery, diapers_and_hygiene
  FOOD_PRODUCTS : food_drink, food, drinks, la_cuisine
  AUTO_PARTS    : auto, construction_tools_*, home_construction
  CHEMICALS     : agro_industry_and_commerce, industry_commerce_and_business
  FURNITURE     : furniture_* (실제 가구류만), office_furniture,
                  kitchen_dining_laundry_garden_furniture
                  ※ bed_bath_table·housewares 제외 → 소형 생활용품이므로 CLOTHING으로 분리
  MACHINERY     : small_appliances, home_appliances, home_appliances_2,
                  small_appliances_home_oven_and_coffee, air_conditioning
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 카테고리 → ItemType 매핑 (영문 번역 기준)
# ---------------------------------------------------------------------------
_CATEGORY_TO_ITEM_TYPE: Dict[str, str] = {}

_MAPPING: Dict[str, list] = {
    "ELECTRONICS": [
        "electronics", "computers", "tablets_printing_image",
        "consoles_games", "audio", "telephony", "fixed_telephony",
        "computers_accessories", "cine_photo",
    ],
    "CLOTHING": [
        "fashion_bags_accessories", "fashion_shoes",
        "fashion_male_clothing", "fashio_female_clothing",
        "fashion_underwear_beach", "fashion_sport",
        "fashion_childrens_clothes", "luggage_accessories",
        # 소형 섬유·생활용품 — 실제 가구류가 아니므로 CLOTHING으로 분류
        "bed_bath_table",   # 침구·수건 등 섬유류 (10,953건, 소형)
        "housewares",       # 소형 주방/생활용품 (6,795건, 소형)
        "home_confort",     # 소형 홈인테리어 소품
        "home_comfort_2",
    ],
    "COSMETICS": [
        "health_beauty", "perfumery", "diapers_and_hygiene",
    ],
    "FOOD_PRODUCTS": [
        "food_drink", "food", "drinks", "la_cuisine",
    ],
    "AUTO_PARTS": [
        "auto",
        "construction_tools_construction",
        "costruction_tools_garden",
        "construction_tools_lights",
        "construction_tools_safety",
        "costruction_tools_tools",
        "home_construction",  # 건축자재류
    ],
    "CHEMICALS": [
        "agro_industry_and_commerce",
        "industry_commerce_and_business",
    ],
    "FURNITURE": [
        # 실제 부피가 큰 가구류만 포함
        "furniture_decor",
        "furniture_mattress_and_upholstery",
        "furniture_living_room",
        "furniture_bedroom",
        "office_furniture",
        "kitchen_dining_laundry_garden_furniture",
    ],
    "MACHINERY": [
        "small_appliances", "home_appliances", "home_appliances_2",
        "small_appliances_home_oven_and_coffee", "air_conditioning",
    ],
}

for _itype, _cats in _MAPPING.items():
    for _cat in _cats:
        _CATEGORY_TO_ITEM_TYPE[_cat] = _itype


# ---------------------------------------------------------------------------
# CBM 스케일 팩터 (Olist 소매 단품 → LCL 상업 화물 단위 보정)
# Olist 중앙값은 개별 상품 치수이므로, LCL 콘솔 화물 단위로 환산하기 위해
# item_type별 경험적 배율을 적용한다.
# ---------------------------------------------------------------------------
_LCL_SCALE: Dict[str, float] = {
    "ELECTRONICS":    72.0,   # 0.0028 → ~0.20 CBM
    "CLOTHING":       30.0,   # ~0.010 (fashion+bed_bath+housewares 혼합 중앙값) → ~0.30 CBM
    "COSMETICS":      53.0,   # 0.0043 → ~0.23 CBM
    "FOOD_PRODUCTS":  127.0,  # 0.0048 → ~0.61 CBM
    "AUTO_PARTS":     168.0,  # 0.0066 → ~1.11 CBM
    "CHEMICALS":      26.0,   # 0.0320 → ~0.83 CBM
    "FURNITURE":      140.0,  # 실제 가구류 중앙값 ~0.023 CBM → ~3.22 CBM (소형 항목 제거 후)
    "MACHINERY":      448.0,  # 0.0100 → ~4.48 CBM
}

# CBM clamp (lo, hi) — LCL 규격 유지
_CBM_CLAMP: Dict[str, Tuple[float, float]] = {
    "ELECTRONICS":    (0.03,  0.60),
    "CLOTHING":       (0.05,  0.90),
    "COSMETICS":      (0.02,  0.40),
    "FOOD_PRODUCTS":  (0.15,  2.00),
    "AUTO_PARTS":     (0.30,  4.00),
    "CHEMICALS":      (0.20,  3.00),
    "FURNITURE":      (0.80,  8.00),
    "MACHINERY":      (1.50, 12.00),
}


# ---------------------------------------------------------------------------
# 내부 계산 함수
# ---------------------------------------------------------------------------

def _compute_rates(archive_dir: Path) -> Tuple[Dict[str, float], float]:
    """
    Olist CSV에서 ItemType별 시간당 도착률을 계산한다.

    Returns
    -------
    raw_rates : {ItemType: shipments_per_hour}  — Olist 원본 스케일
    total_hours : float                          — 데이터 기간(시간)
    """
    try:
        import pandas as pd  # 런타임에만 의존
    except ImportError as exc:
        raise ImportError(
            "olist_calibration 사용에는 pandas가 필요합니다: pip install pandas"
        ) from exc

    orders   = pd.read_csv(archive_dir / "olist_orders_dataset.csv")
    items    = pd.read_csv(archive_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(archive_dir / "olist_products_dataset.csv")
    trans    = pd.read_csv(archive_dir / "product_category_name_translation.csv")

    products = products.merge(trans, on="product_category_name", how="left")
    delivered = orders[orders["order_status"] == "delivered"][
        ["order_id", "order_purchase_timestamp"]
    ]
    df = items.merge(delivered, on="order_id", how="inner")
    df = df.merge(
        products[["product_id", "product_category_name_english",
                  "product_weight_g",
                  "product_length_cm", "product_height_cm", "product_width_cm"]],
        on="product_id", how="left",
    )

    df["item_type"] = (
        df["product_category_name_english"]
        .map(_CATEGORY_TO_ITEM_TYPE)
    )

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    total_hours = (
        df["order_purchase_timestamp"].max()
        - df["order_purchase_timestamp"].min()
    ).total_seconds() / 3600.0

    valid = df.dropna(subset=["item_type"])
    counts = valid["item_type"].value_counts()

    raw_rates = {itype: cnt / total_hours for itype, cnt in counts.items()}
    return raw_rates, total_hours


def _compute_cbm_params(archive_dir: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Olist 실측 CBM 로그 정규 파라미터를 LCL 스케일로 변환한다.

    Returns
    -------
    {ItemType: (mu_log, sigma_log, clamp_lo, clamp_hi)}
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "olist_calibration 사용에는 pandas/numpy가 필요합니다."
        ) from exc

    orders   = pd.read_csv(archive_dir / "olist_orders_dataset.csv")
    items    = pd.read_csv(archive_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(archive_dir / "olist_products_dataset.csv")
    trans    = pd.read_csv(archive_dir / "product_category_name_translation.csv")

    products = products.merge(trans, on="product_category_name", how="left")
    delivered = orders[orders["order_status"] == "delivered"][["order_id"]]
    df = items.merge(delivered, on="order_id", how="inner")
    df = df.merge(
        products[["product_id", "product_category_name_english",
                  "product_length_cm", "product_height_cm", "product_width_cm"]],
        on="product_id", how="left",
    )

    df["cbm_raw"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    ) / 1_000_000.0
    df["item_type"] = df["product_category_name_english"].map(_CATEGORY_TO_ITEM_TYPE)

    df = df.dropna(subset=["item_type", "cbm_raw"])
    df = df[df["cbm_raw"] > 0]

    result: Dict[str, Tuple[float, float, float, float]] = {}
    for itype in _MAPPING:
        g = df[df["item_type"] == itype]["cbm_raw"]
        if len(g) < 10:
            continue
        scale = _LCL_SCALE[itype]
        lcl_cbm = g * scale
        log_vals = np.log(lcl_cbm[lcl_cbm > 0])
        mu    = float(log_vals.mean())
        sigma = float(log_vals.std())
        lo, hi = _CBM_CLAMP[itype]
        result[itype] = (round(mu, 4), round(sigma, 4), lo, hi)

    return result


def load_dimension_samples(
    archive_dir: str | Path,
) -> Dict[str, list[tuple[float, float, float]]]:
    """
    Olist CSV에서 ItemType별 실측 치수 샘플(cm)을 반환한다.

    Returns
    -------
    dict  {ItemType: [(length_cm, height_cm, width_cm), ...]}
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "olist_calibration 사용에는 pandas가 필요합니다: pip install pandas"
        ) from exc

    archive = Path(archive_dir)
    orders = pd.read_csv(archive / "olist_orders_dataset.csv")
    items = pd.read_csv(archive / "olist_order_items_dataset.csv")
    products = pd.read_csv(archive / "olist_products_dataset.csv")
    trans = pd.read_csv(archive / "product_category_name_translation.csv")

    products = products.merge(trans, on="product_category_name", how="left")
    delivered = orders[orders["order_status"] == "delivered"][["order_id"]]
    df = items.merge(delivered, on="order_id", how="inner")
    df = df.merge(
        products[[
            "product_id",
            "product_category_name_english",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]],
        on="product_id",
        how="left",
    )

    df["item_type"] = df["product_category_name_english"].map(_CATEGORY_TO_ITEM_TYPE)
    df = df.dropna(
        subset=[
            "item_type",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    )
    df = df[
        (df["product_length_cm"] > 0)
        & (df["product_height_cm"] > 0)
        & (df["product_width_cm"] > 0)
    ]

    result: Dict[str, list[tuple[float, float, float]]] = {}
    for itype in _MAPPING:
        rows = df[df["item_type"] == itype][[
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]]
        result[itype] = [
            (
                round(float(length), 2),
                round(float(height), 2),
                round(float(width), 2),
            )
            for length, height, width in rows.itertuples(index=False, name=None)
        ]

    return result


def load_destination_samples(
    archive_dir: str | Path,
) -> Dict[str, list[str]]:
    """
    Olist CSV에서 ItemType별 실제 목적지(customer_state) 샘플을 반환한다.

    Returns
    -------
    dict  {ItemType: [customer_state, ...]}
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "olist_calibration 사용에는 pandas가 필요합니다: pip install pandas"
        ) from exc

    archive = Path(archive_dir)
    orders = pd.read_csv(archive / "olist_orders_dataset.csv")
    items = pd.read_csv(archive / "olist_order_items_dataset.csv")
    products = pd.read_csv(archive / "olist_products_dataset.csv")
    customers = pd.read_csv(archive / "olist_customers_dataset.csv")
    trans = pd.read_csv(archive / "product_category_name_translation.csv")

    products = products.merge(trans, on="product_category_name", how="left")
    delivered = orders[orders["order_status"] == "delivered"][["order_id", "customer_id"]]
    df = items.merge(delivered, on="order_id", how="inner")
    df = df.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    df = df.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
    )

    df["item_type"] = df["product_category_name_english"].map(_CATEGORY_TO_ITEM_TYPE)
    df = df.dropna(subset=["item_type", "customer_state"])
    df["customer_state"] = df["customer_state"].astype(str).str.strip().str.upper()
    df = df[df["customer_state"] != ""]

    result: Dict[str, list[str]] = {}
    for itype in _MAPPING:
        states = df[df["item_type"] == itype]["customer_state"].tolist()
        result[itype] = states

    return result


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def load_arrival_rates(
    archive_dir: str | Path,
    total_rate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Olist CSV에서 ItemType별 시간당 도착률을 계산하여 반환한다.

    Parameters
    ----------
    archive_dir : str | Path
        archive 폴더 경로 (olist_*.csv 파일들이 있어야 함).
    total_rate : float, optional
        전체 시간당 총 도착 건수. 지정하면 비율을 유지하면서 스케일 조정.
        None 이면 Olist 원본 스케일(~4.2건/hr) 그대로 반환.

    Returns
    -------
    dict  {ItemType: float}
    """
    raw, _ = _compute_rates(Path(archive_dir))

    if total_rate is None:
        return {k: round(v, 4) for k, v in raw.items()}

    current_total = sum(raw.values())
    scale = total_rate / current_total
    return {k: round(v * scale, 4) for k, v in raw.items()}


def load_cbm_params(
    archive_dir: str | Path,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Olist 실측 치수 기반 CBM 로그 정규 파라미터 (LCL 스케일 변환 후) 반환.

    Returns
    -------
    dict  {ItemType: (mu_log, sigma_log, clamp_lo, clamp_hi)}
    """
    return _compute_cbm_params(Path(archive_dir))


def make_olist_config(
    archive_dir: str | Path,
    total_rate: float = 6.0,
    use_olist_cbm: bool = False,
    **env_kwargs,
):
    """
    Olist 데이터 기반으로 calibrated EnvConfig 를 반환한다.

    Parameters
    ----------
    archive_dir : str | Path
        archive 폴더 경로.
    total_rate : float
        전체 시간당 총 도착 건수 (비율은 Olist 분포 유지).
        기본값 6.0 — 기존 기본 설정과 유사한 규모.
    use_olist_cbm : bool
        True 이면 Olist 실측 치수 → LCL 스케일 CBM 파라미터를 distributions.py에 주입.
        False(기본) 이면 기존 학술 기반 CBM 파라미터 유지.
    **env_kwargs
        EnvConfig 에 추가로 전달할 인자 (seed, sim_duration_hours 등).

    Returns
    -------
    EnvConfig
    """
    from .env import EnvConfig

    rates = load_arrival_rates(archive_dir, total_rate=total_rate)
    cfg = EnvConfig(arrival_rates=rates, **env_kwargs)
    cfg._olist_dimension_samples = load_dimension_samples(archive_dir)
    cfg._olist_destination_samples = load_destination_samples(archive_dir)

    if use_olist_cbm:
        cbm_params = load_cbm_params(archive_dir)
        _inject_cbm_params(cbm_params)

    return cfg


def _inject_cbm_params(
    params: Dict[str, Tuple[float, float, float, float]],
) -> None:
    """distributions._CBM_PARAMS 를 런타임에 교체한다."""
    from . import distributions as _dist
    for itype, p in params.items():
        if itype in _dist._CBM_PARAMS:
            _dist._CBM_PARAMS[itype] = p


# ---------------------------------------------------------------------------
# CLI 요약 출력 (python -m simulator_v1.olist_calibration <archive_dir>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json

    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../archive")
    print("=== Olist 기반 arrival_rates (원본 스케일) ===")
    rates = load_arrival_rates(archive)
    for k, v in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"  {k:<15}: {v:.4f} /hr")

    print(f"\n  합계: {sum(rates.values()):.4f} /hr")

    print("\n=== Olist 기반 CBM 파라미터 (LCL 스케일) ===")
    cbm = load_cbm_params(archive)
    for k, (mu, sigma, lo, hi) in cbm.items():
        median_cbm = round(math.exp(mu), 3)
        print(f"  {k:<15}: mu_log={mu:.3f}  sigma_log={sigma:.3f}  "
              f"median={median_cbm:.3f}  clamp=[{lo}, {hi}]")
