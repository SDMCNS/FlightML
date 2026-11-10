from flightml.data.dataset import FlightDataset
from flightml.weather.metar_client import MetarClient
from flightml.eval.diagnostics import diagnose_flight_bottlenecks


def test_diagnose_flight_bottlenecks():
    ds = FlightDataset()
    client = MetarClient()
    diag = diagnose_flight_bottlenecks(ds, client, 202007026)
    assert diag.ectrl_id == 202007026
    assert diag.adep == "KORD"
    assert diag.ades == "EGLL"
    assert diag.holding_delay_min > 0.0
    assert diag.primary_bottleneck_factor != ""
    assert diag.mitigation_recommendation != ""

