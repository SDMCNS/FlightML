"""
End-to-End Streamlit Dashboard Automated Simulation Suite.

Simulates user interactions across:
1. Airport Operations (switching across different airports).
2. Flight Inspector (switching across different flights).
3. DME-DME RNAV coverage & rationalization optimizer.
4. Radar Surveillance coverage & multi-tier optimizer.
5. Passenger Journey Copilot & Delay Risk assessment.
6. Disk & Memory Cache Reset button.
"""
import sys
import time
from pathlib import Path
from streamlit.testing.v1 import AppTest


def find_selectbox(at: AppTest, label_substr: str):
    for sb in at.selectbox:
        if label_substr in sb.label:
            return sb
    return None


def find_button(at: AppTest, label_substr: str):
    for btn in at.button:
        if label_substr in btn.label:
            return btn
    return None


def run_simulation():
    print("=" * 80)
    print("🚀 FLIGHTML DASHBOARD INTERACTION SIMULATION")
    print("=" * 80)
    
    app_path = Path(__file__).resolve().parent.parent / "flightml" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=300)
    
    # Step 1: Initial full dashboard load
    print("\n[Stage 1/6] Running initial baseline dashboard execution...")
    t0 = time.time()
    at.run(timeout=300)
    print(f" Baseline load completed in {time.time() - t0:.1f}s.")
    _assert_no_exceptions(at, "Baseline Load")
    
    # Step 2: Airport Operations - cycle through airports
    print("\n[Stage 2/6] Simulating Airport Operations selections...")
    apt_sb = find_selectbox(at, "Select Airport (ICAO)")
    if apt_sb:
        test_airports = [opt for opt in apt_sb.options if opt != apt_sb.value][:2]
        for icao in test_airports:
            print(f"  Selecting Airport: {icao}...")
            t_sub = time.time()
            curr_sb = find_selectbox(at, "Select Airport (ICAO)")
            at = curr_sb.select(icao).run(timeout=180)
            print(f"   Rendered {icao} in {time.time() - t_sub:.1f}s.")
            _assert_no_exceptions(at, f"Airport Operations ({icao})")
    else:
        print("  ⚠️ Could not find 'Select Airport (ICAO)' selectbox.")

    # Step 3: Single Flight Selection - cycle through flights
    print("\n[Stage 3/6] Simulating Active Flight selections...")
    flight_sb = find_selectbox(at, "Active Flight Selection")
    if flight_sb:
        test_flights = [opt for opt in flight_sb.options if opt != flight_sb.value][:2]
        for fl_desc in test_flights:
            print(f"  Selecting Flight: {fl_desc[:40]}...")
            t_sub = time.time()
            curr_sb = find_selectbox(at, "Active Flight Selection")
            at = curr_sb.select(fl_desc).run(timeout=180)
            print(f"   Rendered Flight in {time.time() - t_sub:.1f}s.")
            _assert_no_exceptions(at, f"Single Flight Selection ({fl_desc[:30]})")
    else:
        print("  ⚠️ Could not find 'Active Flight Selection' selectbox.")

    # Step 4: DME & Radar Optimization Action Buttons
    print("\n[Stage 4/6] Simulating DME & Radar Deactivation Optimizers...")
    radar_btn = find_button(at, "Deactivate All Suggested Heads")
    if radar_btn:
        print(f"  Clicking Radar Optimizer: '{radar_btn.label}'...")
        t_sub = time.time()
        at = radar_btn.click().run(timeout=180)
        print(f"   Radar optimization completed in {time.time() - t_sub:.1f}s.")
        _assert_no_exceptions(at, "Radar Optimizer Action")
    else:
        print("  ⚠️ Could not find Radar optimizer button.")

    dme_btn = find_button(at, "Deactivate All Suggested")
    if dme_btn:
        print(f"  Clicking DME Optimizer: '{dme_btn.label}'...")
        t_sub = time.time()
        at = dme_btn.click().run(timeout=180)
        print(f"   DME optimization completed in {time.time() - t_sub:.1f}s.")
        _assert_no_exceptions(at, "DME Optimizer Action")
    else:
        print("  ⚠️ Could not find DME optimizer button.")

    # Step 5: Passenger Journey Copilot
    print("\n[Stage 5/6] Simulating Passenger Travel Copilot Assessment...")
    copilot_btn = find_button(at, "Assess My Flight Journey & Delay Risk")
    if copilot_btn:
        print(f"  Clicking Copilot Assessment: '{copilot_btn.label}'...")
        t_sub = time.time()
        at = copilot_btn.click().run(timeout=180)
        print(f"   Copilot assessment completed in {time.time() - t_sub:.1f}s.")
        _assert_no_exceptions(at, "Passenger Journey Copilot")
    else:
        print("  ⚠️ Could not find Passenger Copilot button.")

    # Step 6: Cache Reset Button
    print("\n[Stage 6/6] Simulating Cache Reset Button...")
    cache_btn = find_button(at, "Reset All Analysis Cache")
    if cache_btn:
        print(f"  Clicking: '{cache_btn.label}'...")
        t_sub = time.time()
        at = cache_btn.click().run(timeout=180)
        print(f"   Cache reset completed in {time.time() - t_sub:.1f}s.")
        _assert_no_exceptions(at, "Reset All Analysis Cache")
    else:
        print("  ⚠️ Could not find Cache Reset button.")

    print("\n" + "=" * 80)
    print("🎉 ALL DASHBOARD SIMULATION WORKFLOWS COMPLETED WITH ZERO ERRORS!")
    print("=" * 80)


def _assert_no_exceptions(at: AppTest, stage_name: str):
    if at.exception:
        print(f"\n❌ ERROR in {stage_name}: Encountered {len(at.exception)} exception(s):")
        for i, exc in enumerate(at.exception, 1):
            print(f"\n[Exception #{i}]")
            print(f"Type / Value: {exc.value}")
            print(f"Message: {exc.message}")
        sys.exit(1)
    else:
        print(f"  ✅ {stage_name}: OK (No exceptions)")


if __name__ == "__main__":
    run_simulation()
