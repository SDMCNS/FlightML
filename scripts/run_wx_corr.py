import pandas as pd
from flightml.data.dataset import FlightDataset
from flightml.weather.metar_client import MetarClient

ds = FlightDataset()
client = MetarClient()

hubs = ["EGLL", "LFPG", "EHAM", "EDDF"]
start_date = "2016-11-01"
end_date = "2016-11-10"

traffic_df = ds.query(f"""
    SELECT 
        CAST(actual_off_block_time AS DATE) as date_val,
        count(*) as flights,
        round(avg(dep_delay_min), 2) as avg_dep_delay,
        round(avg(arr_delay_min), 2) as avg_arr_delay,
        round(sum(case when arr_delay_min > 30 then 1 else 0 end) * 100.0 / count(*), 2) as pct_delayed_30
    FROM flights
    WHERE actual_off_block_time >= '{start_date}' AND actual_off_block_time <= '{end_date} 23:59:59'
    GROUP BY CAST(actual_off_block_time AS DATE)
    ORDER BY date_val ASC
""")

traffic_df["date_str"] = traffic_df["date_val"].astype(str)

wx_records = []
for d in pd.date_range(start_date, end_date):
    d_str = d.strftime("%Y-%m-%d")
    daily_ifr_count = 0
    daily_obs_count = 0
    
    for h in hubs:
        df_h = client.fetch_station_metars(h, f"{d_str} 00:00:00", f"{d_str} 23:59:59")
        if not df_h.empty:
            daily_obs_count += len(df_h)
            if "flight_category" in df_h.columns:
                daily_ifr_count += (df_h["flight_category"].isin(["IFR", "LIFR"])).sum()
    
    ifr_pct = (daily_ifr_count / daily_obs_count * 100.0) if daily_obs_count > 0 else 0.0
    wx_records.append({"date_str": d_str, "obs_count": daily_obs_count, "ifr_pct": round(ifr_pct, 1)})

wx_df = pd.DataFrame(wx_records)

merged = pd.merge(traffic_df, wx_df, on="date_str")
print("\nMerged Traffic & Weather:")
print(merged[["date_str", "flights", "avg_arr_delay", "ifr_pct", "pct_delayed_30"]])

corr_delay = merged["avg_arr_delay"].corr(merged["ifr_pct"])
corr_severe = merged["pct_delayed_30"].corr(merged["ifr_pct"])
print(f"\nPearson Correlation (Avg Delay vs IFR %): {corr_delay:.3f}")
print(f"Pearson Correlation (Severe Delays >30m vs IFR %): {corr_severe:.3f}")
