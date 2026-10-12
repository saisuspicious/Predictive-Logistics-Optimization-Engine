# 🚚 Predictive Logistics & Fleet Route Optimization Engine

An end-to-end solo software engineering pipeline that addresses shipping inefficiencies by connecting data-driven trip forecasting models with vehicle route allocation engines.

---

## 🏗️ Technical Pipeline Design
* **Data Processing Layer:** Vectorized geolocational distance analytics executing spatial metrics (Haversine formulations).
* **Machine Learning Regressor:** Tabular Gradient Boosting framework (**LightGBM**) analyzing dynamic transit bottlenecks over 3.9+ million records.
* **Algorithmic Core Solver:** Resolves the Capacitated Vehicle Routing Problem (CVRP) via meta-heuristics using **Google OR-Tools**.
* **Dashboard Control Portal:** Interactive operational telemetry views built with modular **Streamlit** components.

---

## 📊 Core Performance Benchmarks
* **Processing Scaling Profile:** 3,946,698 Input Records handled.
* **Forecasting Accuracy Threshold:** Mean Absolute Error (MAE) = 3.40 minutes.
* **Optimization Dispatch Rate:** Multi-vehicle fleet convergence completed in < 0.05 seconds.

---

## 🚀 Execution Configurations

1. Run the local machine learning engine pipeline:
   ```cmd
   python phase2_train.py
   ```
2. Start the interactive distribution dashboard control platform:
   ```cmd
   python -m streamlit run app_ui.py
   ```
