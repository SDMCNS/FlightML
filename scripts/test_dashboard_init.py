from flightml.dashboard.app import _get_available_periods, get_dataset
import traceback

periods = _get_available_periods()
print("Periods:", periods)
for p in periods:
    try:
        ds = get_dataset(p)
        print(f"Success for period {p}: {ds.con.execute('SELECT count(*) FROM flights').fetchone()[0]} flights")
    except Exception as e:
        print(f"FAILED for period {p}: {e}")
        traceback.print_exc()

