from flightml.data.dataset import FlightDataset
from flightml.models.reactionary_delay import ReactionaryDelayAnalyzer


def test_reactionary_delay_analyzer():
    ds = FlightDataset()
    analyzer = ReactionaryDelayAnalyzer(ds)
    stats = analyzer.compute_network_reactionary_multiplier(
        operator="RYR", date_start="2016-11-01", date_end="2016-11-03"
    )
    assert "total_turnarounds_analyzed" in stats
    assert stats["total_turnarounds_analyzed"] > 0
    assert 0.0 <= stats["delay_propagation_ratio"] <= 1.0
    assert 0.0 <= stats["ground_buffer_absorption_rate"] <= 1.0

