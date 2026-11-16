# FlightML 5-Stage LLM Evaluation Benchmark 📊

Comprehensive critical evaluation comparing **Deterministic Ground Truth (FlightML CLI)**, **Base Qwen 2.5 7B Instruct**, **EuroAvia-7B (Fine-Tuned QLoRA)**, **Qwen 2.5 7B + FastMCP**, and **EuroAvia-7B + FastMCP** across the 5 canonical aviation intelligence test tracks.

## 1. Executive Performance Scorecard

| Test Scenario | Stage 1: Deterministic | Stage 2: Qwen 2.5 Base | Stage 3: EuroAvia-7B | Stage 4: Qwen + MCP | Stage 5: EuroAvia + MCP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test 1: Trajectory Physics & PRU Scorecards** | 100 / 100 (Truth) | 60 / 100 | **76 / 100** | 65 / 100 | **76 / 100** |
| **Test 2: Fuel Burn & Eurocontrol CRCO Navigation Fees (€ EUR)** | 100 / 100 (Truth) | 55 / 100 | **95 / 100** | 50 / 100 | **95 / 100** |
| **Test 3: Rotational Delay Contagion & Tail Tracking** | 100 / 100 (Truth) | 42 / 100 | **85 / 100** | 35 / 100 | **85 / 100** |
| **Test 4: 4D Airspace Closure Counterfactual** | 100 / 100 (Truth) | 80 / 100 | **45 / 100** | 80 / 100 | **65 / 100** |
| **Test 5: Passenger Travel Copilot & Statutory Rights (EU261 / UK261)** | 100 / 100 (Truth) | 48 / 100 | **95 / 100** | 95 / 100 | **95 / 100** |
| **OVERALL COMPOSITE AVERAGE** | **100.0 / 100** | **57.0 / 100** | **79.2 / 100** | **65.0 / 100** | **83.2 / 100** |

## 2. Key Findings & Critical Evaluation

### A. Fine-Tuning Impact (Stage 2 vs Stage 3)
- Base Qwen 2.5 7B averaged **57.0/100**, frequently drifting into generic explanations, approximate numbers, or US Dollar (`$`) conventions.
- EuroAvia-7B (Fine-Tuned) averaged **79.2/100** standalone (+22.2 pts increase), demonstrating internalized knowledge of Eurocontrol PRU KPIs (KEP, KEA, ASMA), CRCO unit rates, EU261 legal thresholds, and strict adherence to Euro (`€`, `EUR`) currency formatting.

### B. The Power of Live MCP Tool Augmentation (Stage 4 & Stage 5)
- When augmented with live FlightML FastMCP tools, base Qwen climbed from 57.0 to **65.0/100**, successfully pulling real telemetry from DuckDB.
- EuroAvia-7B + FastMCP reached the pinnacle at **83.2/100**, combining verified live numerical precision with domain-authoritative prose and regulatory rigor.

## 3. Test Details Index
- [Test 1: Trajectory Physics & PRU Scorecards](file:///C:/Projects/data/FlightML/llm_tests/test_1_test_1_trajectory_physics_&_pru_scorecards.md)
- [Test 2: Fuel Burn & Eurocontrol CRCO Navigation Fees (€ EUR)](file:///C:/Projects/data/FlightML/llm_tests/test_2_test_2_fuel_burn_&_eurocontrol_crco_navigation_fees_eur_eur.md)
- [Test 3: Rotational Delay Contagion & Tail Tracking](file:///C:/Projects/data/FlightML/llm_tests/test_3_test_3_rotational_delay_contagion_&_tail_tracking.md)
- [Test 4: 4D Airspace Closure Counterfactual](file:///C:/Projects/data/FlightML/llm_tests/test_4_test_4_4d_airspace_closure_counterfactual.md)
- [Test 5: Passenger Travel Copilot & Statutory Rights (EU261 / UK261)](file:///C:/Projects/data/FlightML/llm_tests/test_5_test_5_passenger_travel_copilot_&_statutory_rights_eu261___uk261.md)
