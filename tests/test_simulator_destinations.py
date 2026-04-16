from pathlib import Path

from simulator_v1.env import ConsolidationEnv, EnvConfig
from simulator_v1.olist_calibration import load_destination_samples, make_olist_config


def test_generate_arrivals_assigns_configured_destinations():
    cfg = EnvConfig(
        seed=7,
        destination="PORT_FALLBACK",
        destinations=["BUSAN", "OSAKA", "LA"],
        destination_weights={"BUSAN": 0.1, "OSAKA": 0.2, "LA": 0.7},
        use_real_distributions=False,
        arrival_rates={
            "ELECTRONICS": 20.0,
            "CLOTHING": 0.0,
            "COSMETICS": 0.0,
            "FOOD_PRODUCTS": 0.0,
            "AUTO_PARTS": 0.0,
            "CHEMICALS": 0.0,
            "FURNITURE": 0.0,
            "MACHINERY": 0.0,
        },
    )
    env = ConsolidationEnv(cfg)

    arrivals = env._generate_arrivals(0.0)

    assert arrivals
    assert {shipment.destination for shipment in arrivals} <= {"BUSAN", "OSAKA", "LA"}


def test_observation_exposes_destination_for_each_shipment():
    cfg = EnvConfig(
        seed=11,
        destination="PORT_A",
        destinations=["PORT_A", "PORT_B"],
        use_real_distributions=False,
        arrival_rates={
            "ELECTRONICS": 12.0,
            "CLOTHING": 0.0,
            "COSMETICS": 0.0,
            "FOOD_PRODUCTS": 0.0,
            "AUTO_PARTS": 0.0,
            "CHEMICALS": 0.0,
            "FURNITURE": 0.0,
            "MACHINERY": 0.0,
        },
    )
    env = ConsolidationEnv(cfg)
    env._reset()

    for shipment in env._generate_arrivals(0.0):
        env.buffer.add(shipment)
        env.all_shipments.append(shipment)

    obs = env._build_observation().to_dict()

    assert obs["config"]["destinations"] == ["PORT_A", "PORT_B"]
    assert obs["buffer"]["shipments"]
    assert all("destination" in shipment for shipment in obs["buffer"]["shipments"])


def test_load_destination_samples_reads_olist_customer_states():
    samples = load_destination_samples(Path("olist_dataset"))

    assert samples["ELECTRONICS"]
    assert all(len(state) == 2 and state.isupper() for state in samples["ELECTRONICS"][:20])


def test_make_olist_config_uses_olist_destinations_in_env_generation():
    cfg = make_olist_config(
        archive_dir=Path("olist_dataset"),
        total_rate=2.0,
        sim_duration_hours=1,
        use_olist_cbm=False,
        use_real_distributions=False,
        seed=13,
    )
    env = ConsolidationEnv(cfg)

    arrivals = env._generate_arrivals(0.0)

    assert arrivals
    observed_states = {dest for values in cfg._olist_destination_samples.values() for dest in values}
    assert observed_states
    assert all(shipment.destination in observed_states for shipment in arrivals)
