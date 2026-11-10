import duckdb

con = duckdb.connect()

print("Flight_FIRs_Filed Sample:")
print(con.execute("SELECT * FROM 'data/parquet/periods/202510/flight_firs_filed.parquet' LIMIT 5").df())

print("\nActual vs Filed comparison for flight 291262109:")
print(con.execute("""
    SELECT 
        a."FIR ID" as fir_id,
        a."Entry Time" as act_entry,
        a."Exit Time" as act_exit,
        epoch(a."Exit Time" - a."Entry Time")/60.0 as act_dwell_min,
        f."Entry Time" as fld_entry,
        f."Exit Time" as fld_exit,
        epoch(f."Exit Time" - f."Entry Time")/60.0 as fld_dwell_min,
        (epoch(a."Exit Time" - a."Entry Time") - epoch(f."Exit Time" - f."Entry Time"))/60.0 as dwell_diff_min
    FROM 'data/parquet/periods/202510/flight_firs_actual.parquet' a
    LEFT JOIN 'data/parquet/periods/202510/flight_firs_filed.parquet' f
      ON a."ECTRL ID" = f."ECTRL ID" AND a."FIR ID" = f."FIR ID"
    WHERE a."ECTRL ID" = 291262109 AND a."FIR ID" NOT LIKE '%TAXI%'
    ORDER BY a."Sequence Number"
""").df())
