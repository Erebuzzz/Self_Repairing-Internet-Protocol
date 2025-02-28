import threading
import time
import networkx as nx
import random
import dash
from dash import dcc
from dash import html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from collections import defaultdict
import requests
import asyncio
import websockets
import json
import logging
import gzip
import shutil

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Variables
monitoring_active = False
RIPE_RIS_URL = "wss://ris-live.ripe.net/v1/ws/"
legitimate_asns = {12345, 67890}
graph_lock = threading.Lock()
monitor_task = None
topology_task = None  # Track topology update thread
hijacked_prefixes = set()

# Load and preprocess network topology
import networkx as nx
import requests

def load_real_topology():
    """Downloads and loads the AS-733 topology from Network Repository."""
    url = "https://networkrepository.com/data/as-733.graphml"
    filename = "as-733_19981231.graphml"

    # Download the GraphML file if it doesn't exist
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print("✅ AS-733 GraphML file downloaded successfully!")

    # Load the graph
    G = nx.read_graphml(filename)
    print(f"✅ Loaded AS-733 Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    return preprocess_topology(G)

"""def preprocess_topology(G):
    G = nx.convert_node_labels_to_integers(nx.MultiGraph(G))
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k].setdefault("weight", random.randint(1, 10))
        G.edges[u, v, k].setdefault("latency", random.uniform(1, 50))
        G.edges[u, v, k].setdefault("bandwidth", random.randint(1, 100))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G"""

def preprocess_topology(G):
    """Process AS-733 topology by setting default edge attributes."""
    print(f"Before processing: {len(G.nodes())} nodes, {len(G.edges())} edges")

    G = nx.convert_node_labels_to_integers(G)  # Convert labels to integers
    
    for u, v, data in G.edges(data=True):  
        data.setdefault("weight", random.randint(1, 10))
        data.setdefault("latency", random.uniform(1, 50))
        data.setdefault("bandwidth", random.randint(1, 100))

    isolated_nodes = list(nx.isolates(G))
    print(f"Removing {len(isolated_nodes)} isolated nodes")
    G.remove_nodes_from(isolated_nodes)

    print(f"After processing: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G

G = load_real_topology() or nx.erdos_renyi_graph(10, 0.5)

# BGP Monitoring with WebSockets
async def bgp_hijack_monitor():
    global monitoring_active
    while monitoring_active:
        try:
            async with websockets.connect(RIPE_RIS_URL) as ws:
                subscription = {"type": "ris_subscribe", "data": {"type": "UPDATE", "moreSpecific": True}}
                await ws.send(json.dumps(subscription))
                logging.info("🔍 Subscribed to RIPE RIS Live feed")

                while monitoring_active:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(message)
                        prefix = data["data"].get("prefix", "")
                        as_path = data["data"].get("path", [])

                        if detect_hijack(as_path, prefix):
                            logging.warning(f"🚨 BGP Hijack Alert! Prefix {prefix} via AS Path {as_path}")

                    except asyncio.TimeoutError:
                        logging.warning("⚠️ No data received from RIPE RIS Live in 10 seconds.")

        except websockets.exceptions.ConnectionClosedError:
            logging.error("🔴 WebSocket disconnected! Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

def start_bgp_monitor():
    """ Start BGP monitoring safely in a background thread """
    global monitoring_active
    if not monitoring_active:
        monitoring_active = True
        logging.info("✅ BGP Monitoring Started!")

        # ✅ Ensure WebSocket runs in a new event loop
        threading.Thread(target=lambda: asyncio.run(bgp_hijack_monitor()), daemon=True).start()

def stop_bgp_monitor():
    """ Stop the BGP monitoring by setting the flag to False """
    global monitoring_active
    monitoring_active = False

# Dynamic Topology Updates
def update_topology():
    global G
    while monitoring_active:
        time.sleep(5)  # Update interval
        with graph_lock:
            if random.random() < 0.8 and len(G.nodes()) > 1:
                node_to_remove = random.choice(list(G.nodes()))
                G.remove_node(node_to_remove)
                print(f"🔴 Node {node_to_remove} removed")

            if random.random() < 0.3:
                new_node = max(G.nodes(), default=0) + 1
                neighbors = random.sample(list(G.nodes()), min(2, len(G.nodes())))
                G.add_node(new_node)
                for neighbor in neighbors:
                    G.add_edge(new_node, neighbor, weight=random.randint(1, 10), latency=random.uniform(1, 50), bandwidth=random.randint(1, 100))
                print(f"🔄 Node {new_node} added with edges {neighbors}")

# BGP Hijack Detection
def get_historical_routes(prefix):
    response = requests.get(f"https://stat.ripe.net/data/announced-prefixes/data.json?resource={prefix}")
    if response.status_code == 200:
        data = response.json()
        return {item['asn'] for item in data.get("data", {}).get("prefixes", [])}
    return set()

# ✅ Place this helper function here:
def get_prefix_from_as_path(as_path):
    """ Fetch the most common prefix announced by any ASN in the path """
    for asn in as_path:
        response = requests.get(f"https://stat.ripe.net/data/announced-prefixes/data.json?resource=AS{asn}")
        if response.status_code == 200:
            data = response.json()
            prefixes = [item['prefix'] for item in data.get("data", {}).get("prefixes", [])]
            if prefixes:
                print(f"✅ Found historical prefixes for AS{asn}: {prefixes}")
                return prefixes[0]  # Return the first prefix found
    return None

# BGP Hijack Detection
prefix_cache = {}

def detect_hijack(as_path, prefix):
    if not as_path:
        return False

    if not prefix:
        if tuple(as_path) in prefix_cache:
            prefix = prefix_cache[tuple(as_path)]
        else:
            print(f"⚠️ Missing prefix! Attempting to find historical prefixes for AS Path {as_path}")
            prefix = get_prefix_from_as_path(as_path)
            if prefix:
                prefix_cache[tuple(as_path)] = prefix  # Cache it

    historic_routes = get_historical_routes(prefix) if prefix else set()
    last_asn = as_path[-1]

    if last_asn not in legitimate_asns and last_asn not in historic_routes:
        print(f"🚨 Hijack Detected! Unexpected ASN {last_asn} for prefix {prefix if prefix else '[Unknown]'}")
        return True  

    return False

# Generate Graph for Cytoscape
def generate_cytoscape_graph(G):
    """Generates elements for Dash Cytoscape visualization"""
    elements = []  # ✅ Ensure elements is defined at the start

    hijacked_nodes = {str(node) for node in G.nodes() if random.random() < 0.2}  # Simulated hijack detection

    # Add nodes with color classification
    for node in G.nodes():
        color = "red" if str(node) in hijacked_nodes else "blue"
        elements.append({"data": {"id": str(node), "label": str(node)}, "classes": color})

    # Add edges with weights
    for u, v, data in G.edges(data=True):  # ✅ No `keys=True`
        elements.append({
            "data": {
                "source": str(u),
                "target": str(v),
                "weight": data.get("weight", 1)  # Default weight if missing
            }
        })

    return elements  # ✅ Ensure elements is returned

# Dash App Setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Network Monitoring Dashboard"),
    html.Button("Start Monitoring", id="start-button", n_clicks=0),
    html.Button("Stop Monitoring", id="stop-button", n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape-network',
        elements=generate_cytoscape_graph(G),
        layout={'name': 'cose', 'idealEdgeLength': 120, 'nodeRepulsion': 3000},  # Better spacing
        style={'width': '100%', 'height': '700px', 'background-color': '#000000'},  # Dark theme background
        zoomingEnabled=True,
        userZoomingEnabled=True,
        panningEnabled=True,
        userPanningEnabled=True,
        wheelSensitivity=0.5,
        stylesheet=[
            # 🔹 Default Nodes - Gradient Color Based on Degree
            {'selector': 'node', 'style': {
                'width': '12px', 'height': '12px',
                'background-color': 'mapData(degree, 0, 10, #1f78b4, #ff7f00)',  # Blue to Orange Gradient
                'label': 'data(label)',
                'color': '#ffffff', 'font-size': '8px', 'text-outline-width': '1px', 'text-outline-color': '#333333'
            }},
        
            # 🔴 High-Degree Nodes (Hub Nodes)
            {'selector': '[degree >= 8]', 'style': {
                'width': '15px', 'height': '15px',
                'background-color': '#ff0000',  # Red for high-degree nodes
                'border-width': '2px', 'border-color': '#ffff00',  # Yellow border
                'shadow': '5px 5px 5px rgba(255,255,0,0.7)'  # Glow effect
            }},

            # 🔷 Hijacked Nodes (Simulated Warning)
            {'selector': '.red', 'style': {
                'background-color': 'red',
                'width': '14px', 'height': '14px',
                'border-width': '2px', 'border-color': 'yellow',  # Glow for hijacked nodes
                'shadow': '5px 5px 5px rgba(255,255,0,0.7)'
            }},

            # 🔹 Standard Edges - Smooth & Colorful
            {'selector': 'edge', 'style': {
                'width': 'mapData(weight, 1, 10, 1px, 4px)',  # Adjust thickness based on weight
                'line-color': 'mapData(weight, 1, 10, #00ff00, #ff4500)',  # Green to Orange Gradient
                'curve-style': 'bezier',  # Smooth edges
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#ffffff'
            }},

            # 🔥 Important Edges - Stronger Visibility
            {'selector': '[weight >= 8]', 'style': {
                'line-color': '#ff0000',  # Red for high-weight edges
                'width': '5px'
            }}
        ]
    ),

    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

@app.callback(
    [Output("cytoscape-network", "elements"),
     Output("start-button", "disabled"),
     Output("stop-button", "disabled")],
    [Input("start-button", "n_clicks"),
     Input("stop-button", "n_clicks"),
     Input("interval-component", "n_intervals")]
)
def update_network(start_clicks, stop_clicks, n_intervals):
    global monitoring_active, topology_task

    ctx = dash.callback_context
    if not ctx.triggered:
        return generate_cytoscape_graph(G), monitoring_active, not monitoring_active

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "start-button":
        if not monitoring_active:
            monitoring_active = True
            if topology_task is None or not topology_task.is_alive():
                topology_task = threading.Thread(target=update_topology, daemon=True)
                topology_task.start()
            start_bgp_monitor()
            logging.info("✅ Monitoring started!")

    elif trigger_id == "stop-button":
        monitoring_active = False
        stop_bgp_monitor()
        logging.info("⏹ Monitoring stopped.")

    return generate_cytoscape_graph(G), monitoring_active, not monitoring_active

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False, port=8052)