import networkx as nx
import glob
import matplotlib.pyplot as plt
import random
import os
import time
import numpy as np
import threading
import asyncio
import websockets
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
import nest_asyncio
from collections import defaultdict
import requests

RIPE_RIS_URL = "wss://ris-live.ripe.net/v1/ws/"
legitimate_asns = {12345, 67890}

# Q-learning parameters
Q_TABLE = defaultdict(lambda: defaultdict(lambda: 0))
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2

def preprocess_topology(G):
    G = nx.convert_node_labels_to_integers(nx.MultiGraph(G))
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k].setdefault("weight", random.randint(1, 10))
        G.edges[u, v, k].setdefault("latency", random.uniform(1, 50))
        G.edges[u, v, k].setdefault("bandwidth", random.randint(1, 100))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def load_real_topology():
    url = "https://topology-zoo.org/files/TataNld.graphml"  # Example file
    response = requests.get(url)

    if response.status_code == 200:
        with open("TataNld.graphml", "wb") as file:
            file.write(response.content)
    else:
        raise ValueError(f"❌ Error: Could not download file. Status Code: {response.status_code}")

    try:
        G = nx.read_graphml("TataNld.graphml")
        print(f"✅ Loaded topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"❌ Error reading GraphML file: {e}")
        return None

    return preprocess_topology(G) if G else None  # Ensure we return a valid graph

processed_topologies = []
topology = load_real_topology()
if topology:
    processed_topologies.append(topology)

if processed_topologies[0] is None:
    raise ValueError("❌ Error: No valid topology loaded.")

processed_topologies = load_real_topology()
print(f"Preprocessed {len(processed_topologies)} topologies.")

async def bgp_hijack_monitor():
    async with websockets.connect("wss://ris-live.ripe.net/v1/ws/") as ws:
        subscription = {"type": "ris_subscribe", "data": {"type": "UPDATE", "moreSpecific": True}}
        await ws.send(json.dumps(subscription))
        print("📡 Subscribed to RIPE RIS Live for BGP updates...")

        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "data" in data:
                prefix = data["data"].get("prefix", "")
                as_path = data["data"].get("path", [])
                if detect_hijack(as_path):
                    print(f"🚨 BGP Hijack Alert! Prefix {prefix} via AS Path {as_path}")

threading.Thread(target=lambda: asyncio.run(bgp_hijack_monitor()), daemon=True).start()

def detect_hijack(as_path, prefix):
    historic_routes = get_historical_routes(prefix)
    
    if as_path[-1] not in legitimate_asns or set(as_path) not in historic_routes:
        print(f"🚨 Possible BGP Hijack detected for {prefix}")
        return True
    return False

def get_historical_routes(prefix):
    response = requests.get(f"https://stat.ripe.net/data/announced-prefixes/data.json?resource={prefix}")
    if response.status_code == 200:
        data = response.json()
        return {item['asn'] for item in data.get("data", {}).get("prefixes", [])}
    return set()

def handle_hijack_event(prefix, hijacker_as):
    print(f"🚨 BGP Hijack detected: {prefix} hijacked by AS{hijacker_as}")
    reroute_traffic(prefix)

def reroute_traffic(prefix):
    print(f"🔄 Automatically rerouting traffic away from {prefix}")
    # Implement dynamic routing change here

def update_q_table(state, action, reward, next_state):
    best_next_action = max(Q_TABLE[next_state], key=Q_TABLE[next_state].get, default=0)
    Q_TABLE[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * Q_TABLE[next_state][best_next_action] - Q_TABLE[state][action])

def intelligent_routing(G, source, target):
    available_paths = list(nx.all_simple_paths(G, source, target))
    if not available_paths:
        return None

    # Use Q-learning to decide the best path
    rewards = {}
    for path in available_paths:
        rewards[tuple(path)] = sum(G.edges[path[i], path[i+1], 0]['latency'] for i in range(len(path)-1))
    
    if random.random() < EPSILON:
        return random.choice(available_paths)  # Explore

    best_path = min(rewards, key=rewards.get)  # Exploit
    return list(best_path)
    
    if random.random() < EPSILON:
        return random.choice(available_paths)
    
    best_path = min(available_paths, key=lambda path: sum(G.edges[path[i], path[i+1], 0]['latency'] for i in range(len(path)-1)))
    return best_path

def dynamic_qos_routing(G, source, target):
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k]["latency"] = random.uniform(1, 50)  # Simulating real-time updates

    try:
        path = nx.shortest_path(G, source, target, weight="latency")
        print(f"⚡ Real-time QoS path: {path}")
        return path
    except nx.NetworkXNoPath:
        print("❌ No available path based on QoS metrics.")
        return None

def monitor_network(G, interval=5):
    global monitoring_active  # Ensure we're using the global variable
    while monitoring_active:  # Continue only while monitoring is active
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔍 {timestamp}: Monitoring network...")

        dynamic_qos_routing(G, 0, max(G.nodes()))
        update_topology(G)
        
        time.sleep(interval)  # Pause before next iteration
    def fetch_ripe_atlas_data():
        response = requests.get("https://atlas.ripe.net/api/v2/measurements/5001/")
        if response.status_code == 200:
            data = response.json()
            print("🌍 RIPE Atlas Network Measurement:", data["description"])
        else:
            print("⚠️ Failed to fetch RIPE Atlas data.")

    # Call this inside the monitoring loop
    fetch_ripe_atlas_data()

    print("⏹ Monitoring stopped.")  # Notify when monitoring stops

    def recover_from_failure(G, failed_node):
        """Dynamically repair the network by rerouting traffic."""
        print(f"⚠️ Node failure detected: {failed_node}. Finding alternative paths...")
        for neighbor in list(G.neighbors(failed_node)):
            alternative_paths = list(nx.all_simple_paths(G, source=neighbor, target=max(G.nodes())))
            if alternative_paths:
                best_path = min(alternative_paths, key=lambda path: sum(G.edges[path[i], path[i+1], 0]['latency'] for i in range(len(path)-1)))
                print(f"🔄 Traffic rerouted via {best_path}")
                return best_path
        print("❌ No alternative path found. Network might be disrupted.")
        return None
    

def start_bgp_hijack_monitor():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bgp_hijack_monitor())

app = dash.Dash(__name__)

def update_topology(G):
    if random.random() < 0.3:
        node_to_remove = random.choice(list(G.nodes()))
        G.remove_node(node_to_remove)
        print(f"🔄 BGP Update: Removed node {node_to_remove}")

    if random.random() < 0.3:
        new_node = max(G.nodes()) + 1
        potential_neighbors = list(G.nodes())
        random.shuffle(potential_neighbors)
        
        for neighbor in potential_neighbors[:2]:
            G.add_edge(new_node, neighbor, weight=random.randint(1, 10), latency=random.uniform(1, 50), bandwidth=random.randint(1, 100))
        print(f"🔄 BGP Update: Added node {new_node} with links to {potential_neighbors[:2]}")

    for node in list(G.nodes()):
        if random.random() < 0.2:
            neighbors = list(G.neighbors(node))
            if neighbors:
                withdrawn_neighbor = random.choice(neighbors)
                G.remove_edge(node, withdrawn_neighbor)
                print(f"📉 BGP Route Withdrawal: Node {node} lost link to {withdrawn_neighbor}")

def generate_cytoscape_graph(G):
    elements = [
        {"data": {"id": str(node), "label": str(node)}, "classes": "failed-node" if G.degree(node) == 0 else ""}
        for node in G.nodes()
    ]
    elements += [
        {"data": {"source": str(u), "target": str(v), "weight": G.edges[u, v]["weight"]}}
        for u, v in G.edges()
    ]
    return elements

    print("🔍 Checking Edge List:")
    print(list(processed_topologies[0].edges(data=True))[:5])  # Print first 5 edges


app.layout = html.Div([
    html.H1("Network Monitoring Dashboard"),
    html.Button("Start Monitoring", id="start-button", n_clicks=0),
    html.Button("Stop Monitoring", id="stop-button", n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape-network',
        elements=generate_cytoscape_graph(processed_topologies[0]),
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'}
    ),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

monitoring_active = False  # Default state: Monitoring is OFF

@app.callback(
    Output("cytoscape-network", "elements"),
    [Input("start-button", "n_clicks"),
     Input("stop-button", "n_clicks"),
     Input("interval-component", "n_intervals")]
)
def update_network(start_clicks, stop_clicks, n_intervals):
    global monitoring_active  # Use the global variable

    ctx = dash.callback_context
    if not ctx.triggered:
        return generate_cytoscape_graph(processed_topologies[0])

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "start-button":
        if not monitoring_active:
            monitoring_active = True
            threading.Thread(target=monitor_network, args=(processed_topologies[0],), daemon=True).start()
            print("✅ Monitoring started!")

    elif trigger_id == "stop-button":
        monitoring_active = False  # Stop monitoring
        print("⏹ Monitoring stopped.")

    return generate_cytoscape_graph(processed_topologies[0])

if __name__== "__main__":
    threading.Thread(target=start_bgp_hijack_monitor, daemon=True).start()
    app.run_server(debug=True, use_reloader=False, port=8051)