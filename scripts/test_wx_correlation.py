from flightml.data.dataset import FlightDataset
from flightml.weather.metar_client import MetarClient
import pandas as pd

ds = FlightDataset()
client = MetarClient()

# Check daily flight delay aggregation
daily_flights = ds.query("""
    SELECT 
        CAST(actual_off_block_time AS DATE) as flight_date,
        count(*) as flight_count,
        avg(dep_delay_min) as avg_dep_delay,
        avg(arr_delay_min) as avg_arr_delay,
        sum(case when arr_delay_min > 30 then 1 else 0 end) * 100.0 / count(*) as pct_delayed_30
    FROM flights
    WHERE actual_off_block_time IS NOT NULL
    GROUP BY CAST(actual_off_block_time AS DATE)
    ORDER BY flight_date ASC
""")
print("Daily Flights Sample:")
print(daily_flights.head(5))

# Check METAR for EGLL on first 3 days
for d in ['2016-11-01', '2016-11-02', '2016-11-03']:
    df_wx = client.fetch_station_archive('EGLL', start_date=f'{d} 00:00:00', end_date=f'{d} 23:59:59')
    print(f"EGLL {d} METAR rows: {len(df_wx)}")

