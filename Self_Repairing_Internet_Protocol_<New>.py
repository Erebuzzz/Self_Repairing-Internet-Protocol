import threading
import time
import networkx as nx
import random
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from collections import defaultdict
import requests
import asyncio
import websockets
import json

# Global Variables
monitoring_active = False
RIPE_RIS_URL = "wss://ris-live.ripe.net/v1/ws/"
legitimate_asns = {12345, 67890}
graph_lock = threading.Lock()
monitor_task = None
topology_task = None  # Track topology update thread

# Load and preprocess network topology
def load_real_topology():
    url = "https://topology-zoo.org/files/TataNld.graphml"
    response = requests.get(url)
    if response.status_code == 200:
        with open("TataNld.graphml", "wb") as file:
            file.write(response.content)
        G = nx.read_graphml("TataNld.graphml")
        return preprocess_topology(G)
    return None

def preprocess_topology(G):
    G = nx.convert_node_labels_to_integers(nx.MultiGraph(G))
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k].setdefault("weight", random.randint(1, 10))
        G.edges[u, v, k].setdefault("latency", random.uniform(1, 50))
        G.edges[u, v, k].setdefault("bandwidth", random.randint(1, 100))
    G.remove_nodes_from(list(nx.isolates(G)))
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
                print("🔍 Subscribed to RIPE RIS Live feed")

                while monitoring_active:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(message)
                        prefix = data["data"].get("prefix", None)
                        as_path = data["data"].get("path", [])

                        if detect_hijack(as_path, prefix):
                            print(f"🚨 BGP Hijack Alert! Prefix {prefix if prefix else '[Unknown]'} via AS Path {as_path}")

                    except asyncio.TimeoutError:
                        print("⚠️ No data received from RIPE RIS Live in 10 seconds.")

        except websockets.exceptions.ConnectionClosedError:
            print("🔴 WebSocket disconnected! Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

def start_bgp_monitor():
    """ Start BGP monitoring safely in a background thread """
    global monitoring_active
    if not monitoring_active:
        monitoring_active = True
        print("✅ BGP Monitoring Started!")

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
    elements = [{"data": {"id": str(node), "label": str(node)}} for node in G.nodes()]
    
    valid_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):  # ✅ Get key and data properly
        valid_edges.append({
            "data": {
                "source": str(u),
                "target": str(v),
                "weight": data.get("weight", 1)
            }
        })
    
    elements.extend(valid_edges)
    return elements

# Dash App Setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Network Monitoring Dashboard"),
    html.Button("Start Monitoring", id="start-button", n_clicks=0),
    html.Button("Stop Monitoring", id="stop-button", n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape-network',
        elements=generate_cytoscape_graph(G),
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'}
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
            print("✅ Monitoring started!")

    elif trigger_id == "stop-button":
        monitoring_active = False
        stop_bgp_monitor()
        print("⏹ Monitoring stopped.")

    return generate_cytoscape_graph(G), monitoring_active, not monitoring_active

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False, port=8051)