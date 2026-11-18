"""
Simulate Streamlit dashboard execution using Streamlit AppTest.
Captures any uncaught exceptions, widget errors, or rendering failures.
"""
import sys
import time
from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dashboard_simulation():
    print("=" * 70)
    print("Starting Streamlit Dashboard Simulation (AppTest)...")
    print("=" * 70)
    t0 = time.time()
    
    app_path = Path(__file__).resolve().parent.parent / "flightml" / "dashboard" / "app.py"
    print(f"Target App Path: {app_path}")
    
    # Initialize AppTest pointing to the main dashboard
    at = AppTest.from_file(str(app_path), default_timeout=300)
    
    # Run the initial full dashboard load
    print("Executing full initial script run...")
    at.run(timeout=300)
    
    elapsed = time.time() - t0
    print(f"Initial run completed in {elapsed:.1f}s.")
    
    # Check for exceptions
    exceptions = at.exception
    if exceptions:
        print(f"\n❌ FOUND {len(exceptions)} UNCAUGHT EXCEPTION(S):")
        for i, exc in enumerate(exceptions, 1):
            print(f"\n--- Exception #{i} ---")
            print(f"Type / Message: {exc.value}")
            print(f"Details: {exc.message}")
        sys.exit(1)
    else:
        print("\n✅ Zero uncaught exceptions on initial full run!")
        
    print(f"Total elements rendered: {len(at)}")
    print(f"Tabs rendered: {len(at.tabs)}")
    print(f"Dataframes rendered: {len(at.dataframe)}")
    print(f"Metrics rendered: {len(at.metric)}")
    print(f"Selectboxes rendered: {len(at.selectbox)}")
    print("=" * 70)

if __name__ == "__main__":
    test_dashboard_simulation()
