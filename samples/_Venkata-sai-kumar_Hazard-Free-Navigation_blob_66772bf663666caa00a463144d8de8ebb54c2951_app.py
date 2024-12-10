from flask import Flask, render_template, request, jsonify
import osmnx as ox
import networkx as nx
import json
import heapq
import pandas as pd
import math
import pickle
from shapely.geometry import LineString
import openai

app = Flask(__name__)


# def compute_and_save_graph(location, filename):
#     G = ox.graph_from_place(location, network_type='drive')
#     with open(filename, 'wb') as file:
#         pickle.dump(G, file)

# compute_and_save_graph('San Francisco, California, USA', 'san_francisco_graph.pkl')

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

G = load_graph('san_francisco_graph.pkl')

# def create_crime_data_json():
#     crime_df = pd.read_csv('hotspots.csv')  

#     # Extracting the coordinates from the DataFrame
#     crime_data = crime_df[['Latitude', 'Longitude']].to_dict(orient='records')
#     #limited_data = crime_data[:100] 

#     # Saving the coordinates to a JSON file
#     with open('static/crime_data.json', 'w') as json_file:
#         json.dump(crime_data, json_file, indent=4)

# create_crime_data_json() 

def load_crime_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def euclidean_distance(G, u, v):
    u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
    v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
    return math.sqrt((u_x - v_x) ** 2 + (u_y - v_y) ** 2)

def astar_path(G, source, target, weight='length'):
    # Created a priority queue and hash set to store the visited nodes
    pq = []
    heapq.heappush(pq, (0, source, 0))
    visited = set()
    # Dictionary for the distances
    distances = {node: float('infinity') for node in G.nodes}
    distances[source] = 0
    # Dictionary for predecessors
    predecessors = {node: None for node in G.nodes}

    while pq:
        current_distance, current_node, current_estimated = heapq.heappop(pq)

        if current_node == target:
            break

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor].get(weight, 1)
            distance_through_node = current_distance + edge_weight
            estimated_distance = distance_through_node + euclidean_distance(G, neighbor, target)

            if distance_through_node < distances[neighbor]:
                distances[neighbor] = distance_through_node
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (estimated_distance, neighbor, distance_through_node))

    # Shortest path
    path = []
    while target is not None:
        path.append(target)
        target = predecessors[target]
    path.reverse()

    return path


def dijkstra_path(G, source, target, weight='length'):
    # Creating  a priority queue and hash set to store the visited nodes
    pq = []
    heapq.heappush(pq, (0, source))
    visited = set()
    # Dictionary for the distances
    distances = {node: float('infinity') for node in G.nodes}
    distances[source] = 0
    # Dictionary for predecessor nodes
    predecessors = {node: None for node in G.nodes}

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_node == target:
            break

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor].get(weight, 1)
            distance_through_node = current_distance + edge_weight

            if distance_through_node < distances[neighbor]:
                distances[neighbor] = distance_through_node
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance_through_node, neighbor))

    # Shortest path
    path = []
    while target is not None:
        path.append(target)
        target = predecessors[target]
    path.reverse()

    return path

def calculate_paths(source_location, destination_location, crime_data,algorithm):
    source_coords = ox.geocode(source_location + ', San Francisco, California, USA')
    destination_coords = ox.geocode(destination_location + ', San Francisco, California, USA')
    G = ox.graph_from_place('San Francisco, California, USA', network_type='drive')

    # Converting the crime data coordinates into nodes
    crime_nodes = set()
    for crime in crime_data:
        crime_coords = (crime['Latitude'], crime['Longitude'])
        crime_node = ox.distance.nearest_nodes(G, crime_coords[1], crime_coords[0])
        crime_nodes.add(crime_node)

    source_node = ox.distance.nearest_nodes(G, source_coords[1], source_coords[0])
    destination_node = ox.distance.nearest_nodes(G, destination_coords[1], destination_coords[0])

    # path_algo = dijkstra_path if algorithm == 'dijkstra' else astar_path
    if algorithm=='dijkstra':
        path_algo=dijkstra_path
    elif algorithm=='astar_path':
        path_algo=astar_path
    else:
        path_algo=nx.shortest_path

    # Calculating the initial path
    shortest_path_before = path_algo(G, source_node, destination_node, weight='length')

    # Modified graph
    G_modified = G.copy()
    G_modified.remove_nodes_from(crime_nodes)
    shortest_path_after = path_algo(G_modified, source_node, destination_node, weight='length')

    return {
        'path_before': [[G.nodes[node]['y'], G.nodes[node]['x']] for node in shortest_path_before],
        'path_after': [[G_modified.nodes[node]['y'], G_modified.nodes[node]['x']] for node in shortest_path_after],
        'crime_hotspots': crime_data
    }


# Set your OpenAI API key here
openai.api_key = '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'

def query_gpt(prompt):
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=100  # Adjust based on your needs
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)  # Handle errors appropriately in your application

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/find_path', methods=['POST'])
def find_path():
    source = request.form.get('source')
    destination = request.form.get('destination')
    algorithm = request.form.get('algorithm', 'nx.shortest_path')
    crime_data = load_crime_data('static/crime_data.json')
    paths = calculate_paths(source, destination, crime_data,algorithm)
    return jsonify(paths)

@app.route('/ask_gpt', methods=['POST'])
def ask_gpt():
    data = request.get_json()
    prompt = data.get('prompt')
    response = query_gpt(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

## Reference:- https://stackoverflow.com/questions/63690631/osmnx-shortest-path-how-to-skip-node-if-not-reachable-and-take-the-next-neares