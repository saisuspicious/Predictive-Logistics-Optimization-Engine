import joblib
import numpy as np
from ortools.constraint_solver import pywrapcp as routing_pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# 1. Load the predictive machine learning model we generated in Phase 2
model = joblib.load("delivery_time_model.pkl")

def create_predictive_time_matrix(locations, hour=14, is_weekend=0):
    """
    Uses the trained LightGBM model to build an operational time matrix 
    between all delivery points.
    """
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                # 1. Calculate a mock travel distance between coordinate points
                lat1, lon1 = locations[i]
                lat2, lon2 = locations[j]
                # Simulating a distance approximation based on coordinates
                approx_dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0
                
                # 2. Build the exact feature structure your Phase 2 model expects
                # features order matches: ['trip_distance', 'pickup_hour', 'is_weekend', 'passenger_count']
                input_data = [[approx_dist, hour, is_weekend, 1]]
                
                # 3. Predict travel time in minutes using your trained model
                predicted_time_min = model.predict(input_data)[0]
                
                # Convert to integer seconds for the Google OR-Tools routing engine
                matrix[i][j] = int(max(predicted_time_min, 1.0) * 60)
                
    return matrix

def solve_vehicle_routing(time_matrix, num_vehicles=2, depot=0):
    """Calculates the best possible routes using Google OR-Tools optimization solvers."""
    manager = routing_pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot)
    routing = routing_pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Use professional academic heuristics to find the optimal path matrix
    search_parameters = routing_pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    
    routes = {}
    if solution:
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes[f"Van_{vehicle_id + 1}"] = route
    return routes

if __name__ == "__main__":
    # Mock Coordinates: [Latitude, Longitude] 
    # Index 0 is the Warehouse Depot, Indices 1-4 are Customer addresses
    mock_addresses = [
        [40.7128, -74.0060],  # Depot Warehouse
        [40.7306, -73.9352],  # Dropoff 1
        [40.7589, -73.9851],  # Dropoff 2
        [40.7061, -73.9969],  # Dropoff 3
        [40.7484, -73.9857]   # Dropoff 4
    ]
    
    print("⏳ Building predictive time matrix from machine learning outputs...")
    t_matrix = create_predictive_time_matrix(mock_addresses)
    
    print("🧮 Calculating optimal distribution pathways across fleets...")
    final_itinerary = solve_vehicle_routing(t_matrix, num_vehicles=2)
    
    print("\n🎯 Optimized Dispatch Route Matrix Plan:")
    for van, path in final_itinerary.items():
        print(f"👉 {van}: Stop Sequence Map -> {path}")