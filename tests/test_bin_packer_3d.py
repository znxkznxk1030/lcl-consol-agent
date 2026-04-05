from agents.ai_agent.tools.bin_packer_3d import BinPacker3D


def make_shipment(shipment_id: str, **overrides):
    shipment = {
        "shipment_id": shipment_id,
        "item_type": "ELECTRONICS",
        "cargo_category": "GENERAL",
        "cbm": 0.5,
        "weight": 40.0,
        "length_cm": 100.0,
        "width_cm": 100.0,
        "height_cm": 50.0,
    }
    shipment.update(overrides)
    return shipment


def find_position(plan, shipment_id: str):
    for position in plan.positions:
        if position.shipment_id == shipment_id:
            return position
    raise AssertionError(f"shipment not placed: {shipment_id}")


def test_pack_places_multiple_general_shipments():
    packer = BinPacker3D()
    shipments = [
        make_shipment("S1", cbm=0.6, weight=45.0, length_cm=120.0, width_cm=100.0, height_cm=50.0),
        make_shipment("S2", cbm=0.4, weight=30.0, length_cm=80.0, width_cm=100.0, height_cm=50.0),
    ]

    plan = packer.pack(shipments, container_type="40GP", mbl_id="TEST-001")

    assert sorted(p.shipment_id for p in plan.positions) == ["S1", "S2"]
    assert plan.unplaceable_shipments == []
    assert plan.volume_utilization_pct > 0
    assert plan.weight_utilization_pct > 0


def test_pack_respects_hazmat_and_food_zone_limits():
    packer = BinPacker3D()
    shipments = [
        make_shipment(
            "HZ1",
            item_type="CHEMICALS",
            cargo_category="HAZMAT",
            cbm=0.6,
            weight=80.0,
            length_cm=120.0,
            width_cm=100.0,
            height_cm=50.0,
        ),
        make_shipment(
            "FD1",
            item_type="FOOD_PRODUCTS",
            cargo_category="FOOD",
            cbm=0.6,
            weight=60.0,
            length_cm=120.0,
            width_cm=100.0,
            height_cm=50.0,
        ),
    ]

    plan = packer.pack(shipments, container_type="40GP", mbl_id="TEST-HAZMAT-FOOD")

    hazmat = find_position(plan, "HZ1")
    food = find_position(plan, "FD1")
    hazmat_limit = int(12032 * 0.20)

    assert hazmat.x_mm >= 0
    assert hazmat.x_mm + hazmat.length_mm <= hazmat_limit
    assert food.x_mm >= hazmat_limit


def test_pack_rejects_heavy_item_stacked_on_fragile():
    packer = BinPacker3D()
    shipments = [
        make_shipment(
            "FR1",
            cargo_category="FRAGILE",
            weight=30.0,
            cbm=12.76,
            length_cm=580.0,
            width_cm=220.0,
            height_cm=100.0,
        ),
        make_shipment(
            "HV1",
            cargo_category="FRAGILE",
            weight=60.0,
            cbm=1.0,
            length_cm=100.0,
            width_cm=100.0,
            height_cm=100.0,
        ),
    ]

    plan = packer.pack(shipments, container_type="20GP", mbl_id="TEST-FRAGILE")

    assert sorted(p.shipment_id for p in plan.positions) == ["FR1"]
    assert plan.unplaceable_shipments == ["HV1"]


def test_pack_marks_oversized_item_as_unplaceable_when_it_exceeds_container():
    packer = BinPacker3D()
    shipments = [
        make_shipment(
            "OV1",
            item_type="MACHINERY",
            cargo_category="OVERSIZED",
            cbm=23.1,
            weight=3000.0,
            length_cm=700.0,
            width_cm=240.0,
            height_cm=150.0,
        ),
    ]

    plan = packer.pack(shipments, container_type="20GP", mbl_id="TEST-OVERSIZED")

    assert plan.positions == []
    assert plan.unplaceable_shipments == ["OV1"]
