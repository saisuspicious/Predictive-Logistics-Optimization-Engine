import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ortools.constraint_solver import pywrapcp as routing_pywrapcp
from ortools.constraint_solver import routing_enums_pb2

st.set_page_config(page_title="AI Logistics Routing Engine", layout="wide")
st.title("🚚 Predictive Fleet Management & Route Optimization Dashboard")

# 1. Load your trained Phase 2 machine learning predictive model archive
@st.cache_resource
def load_ml_model():
    return joblib.load("delivery_time_model.pkl")

model = load_ml_model()

# 2. Re-import your core routing and timing math engines from Phase 3
def calculate_optimized_schedule(locations, num_vans):
    n = len(locations)
    time_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = locations[i]
                lat2, lon2 = locations[j]
                approx_dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0
                # Input parameters matching model structure features
                predicted_time = model.predict([[approx_dist, 14, 0, 1]])[0]
                time_matrix[i][j] = int(max(predicted_time, 1.0) * 60)
                
    manager = routing_pywrapcp.RoutingIndexManager(len(time_matrix), num_vans, 0)
    routing = routing_pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = routing_pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    
    routes = {}
    if solution:
        for vehicle_id in range(num_vans):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes[f"Delivery Van {vehicle_id + 1}"] = route
    return routes

# 3. Create interactive sidebar elements for the web dashboard controls
st.sidebar.header("🕹️ Dispatch System Controls")
fleet_size = st.sidebar.slider("Number of Delivery Vans in Fleet", min_value=1, max_value=5, value=2)

# Generate mock geographic dropoff locations
mock_coordinates = [
    [40.7128, -74.0060],  # Warehouse Central Depot Location
    [40.7306, -73.9352],  # Customer Drop-off Point 1
    [40.7589, -73.9851],  # Customer Drop-off Point 2
    [40.7061, -73.9969],  # Customer Drop-off Point 3
    [40.7484, -73.9857]   # Customer Drop-off Point 4
]

if st.button("🚀 Calculate Optimized Fleet Distribution Routes"):
    with st.spinner("Processing geospatial telemetry vectors..."):
        itinerary = calculate_optimized_schedule(mock_coordinates, fleet_size)
        
        st.subheader("🎯 Calculated Dispatch Results Matrix")
        for van, route_sequence in itinerary.items():
            st.markdown(f"**{van} Run Itinerary:**")
            st.info(f"Stop Order Sequence ➔ {route_sequence}")
            
        # Draw interactive geolocation coordinates directly onto a dashboard map component
        map_df = pd.DataFrame(mock_coordinates, columns=['lat', 'lon'])
        st.subheader("🗺️ Regional Location Distribution Map Visualization")
        st.map(map_df)