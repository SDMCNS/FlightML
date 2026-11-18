# FlightML TypeScript Cockpit Frontend

A fast, modern, reactive Single-Page Application (SPA) built with **React 19**, **TypeScript**, **Vite**, **Leaflet**, and **Lucide Icons** for interactive air navigation intelligence.

Designed as a non-clunky, zero-reload alternative to Streamlit for tactical flight tracking, GNSS interference monitoring, and passenger journey predictions.

---

## 🚀 Quick Start

### 1. Development Mode (Vite Hot-Reload)
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

### 2. Standalone Production Build
```bash
cd frontend
npm run build
npm run preview
```

### 3. Unified FastAPI Hosting
When built, the production bundle (`frontend/dist`) is automatically mounted by the FlightML FastAPI backend:
```bash
# Terminal 1: Launch FastAPI
python -m uvicorn flightml.api.app:app --port 8000 --reload
```
Then navigate to **[http://localhost:8000/cockpit](http://localhost:8000/cockpit)**.

---

## 🛠️ Key Modules & Navigation

| Module | Icon | Description |
| :--- | :--- | :--- |
| **4D Flight Explorer** | 🧭 | Responsive flight roster list with instant search (`⌘K`), interactive Leaflet route map, and altitude energy strip chart. |
| **GNSS Jamming Shield** | 🛰️ | Real-time electronic warfare corridor filters, **50 NM terminal buffer gate**, and interactive confirmation threshold slider ($\ge 75$). |
| **Airspace Operations & FIR Saturation** | 🌐 | Real-time capacity utilization, en-route sector bottlenecks, and Eurocontrol PRU KEA horizontal directness indicators. |
| **Contrail & Climate Lab** | 🌿 | Schmidt-Appleman Criterion ($T_{SAC}$), Ice Supersaturated Regions ($RHi \ge 100\%$), and eco-minimal vertical reroute trade-offs. |
| **Passenger Travel Copilot** | 🧳 | Multi-leg connection risk analysis, Minimum Connecting Time (MCT) buffers, and statutory EU261 / UK261 compensation entitlements. |
| **Analytical Math & Proofs** | 📐 | Searchable KaTeX mathematical formulations, physical boundary constraints, and worked verification proofs. |

---

## ⚡ Architecture Highlights
- **Zero Script Re-runs:** Pure client-side reactive state eliminates Streamlit's layout shifts and sluggish widget re-evaluations.
- **Graceful Offline Fallback:** If the backend API (`localhost:8000`) is offline, the cockpit automatically operates in high-fidelity standalone demo mode with authentic Eurocontrol sample flights.
