import pandas as pd
df = pd.read_csv('bank-full.csv',sep = ";")

import networkx as nx

#  Imports
import pandas as pd
import networkx as nx
import numpy as np
from dotenv import load_dotenv
import os
from groq import Groq

#  Load dataset
df = pd.read_csv("bank-full.csv", sep=";")

#  Normalize duration
df['norm_duration'] = (df['duration'] - df['duration'].min()) / (df['duration'].max() - df['duration'].min() + 1e-5)

#  Create Directed Graph
G = nx.DiGraph()

#  Build graph: job → contact → poutcome → y
for _, row in df.iterrows():
    path = [row['job'], row['contact'], row['poutcome'], row['y']]
    
    # Skip invalid data
    if any(pd.isna(p) or str(p).strip().lower() == "unknown" for p in path):
        continue

    # Skip logically invalid transitions: failure → yes
    if row['poutcome'].strip().lower() == 'failure' and row['y'].strip().lower() == 'yes':
        continue

    path = [str(p).strip() for p in path]

    for i in range(len(path) - 1):
        src, tgt = path[i], path[i + 1]
        if G.has_edge(src, tgt):
            G[src][tgt]['count'] += 1
            G[src][tgt]['total_delay'] += row['norm_duration']
        else:
            G.add_edge(src, tgt, count=1, total_delay=row['norm_duration'])


#  Finalize cost and delay
for u, v in G.edges:
    edge = G[u][v]
    edge['delay'] = edge['total_delay'] / edge['count']
    edge['cost'] = 1 / edge['count']  # Infrequent paths → higher cost
    del edge['total_delay']
    del edge['count']

print(" Graph created with cost and delay.")

#  Debug node list
print("\nAvailable nodes in graph:")
print(list(G.nodes))

#  Source and destination node
source_node = "admin."    # Confirm this exists
destination_node = "yes"
delay_bound = 0.7

assert source_node in G.nodes, f"Source node '{source_node}' not in graph!"
assert destination_node in G.nodes, f"Destination node '{destination_node}' not in graph!"

# Path evaluation function
def evaluate_path(G, path):
    total_cost = 0
    total_delay = 0
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        total_cost += edge['cost']
        total_delay += edge['delay']
    return total_cost, total_delay

# Find feasible paths
def find_feasible_paths(G, source, target, bound):
    all_paths = list(nx.all_simple_paths(G, source, target))
    return [p for p in all_paths if evaluate_path(G, p)[1] <= bound]

# Query Groq LLM
def query_llm_for_best_path(paths, G, client):
    prompt = "You are a marketing strategist. Evaluate customer journey paths based on cost and delay.\n\n"
    for i, path in enumerate(paths):
        cost, delay = evaluate_path(G, path)
        prompt += f"{i+1}. Path: {' → '.join(path)} | Cost: {cost:.3f} | Delay: {delay:.3f}\n"
    prompt += "\nChoose the most efficient path (low cost & delay). Respond with the number only."

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        idx = int(response.choices[0].message.content.strip()) - 1
        return paths[idx]
    except Exception:
        return paths[0] if paths else None

# Load Groq API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in environment variables.")

client = Groq(api_key=groq_api_key)

# Run pathfinding
paths = find_feasible_paths(G, source_node, destination_node, delay_bound)

if not paths:
    print("No feasible paths found within the delay bound.")
else:
    # Debug print all candidate paths
    print("\nFeasible paths:")
    for i, path in enumerate(paths):
        cost, delay = evaluate_path(G, path)
        print(f"{i+1}. {' → '.join(path)} | Cost: {cost:.3f}, Delay: {delay:.3f}")

    best_path = query_llm_for_best_path(paths, G, client)
    cost, delay = evaluate_path(G, best_path)

    print("\nBest Path Suggested by LLM:")
    print(" → ".join(best_path))
    print(f"Total Cost: {cost:.3f}, Total Delay: {delay:.3f}")


import matplotlib.pyplot as plt

# Position nodes using spring layout for better separation
pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

# Draw nodes
plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue")

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=9)

# Prepare edge attributes for drawing
edge_colors = [G[u][v]['delay'] for u, v in G.edges()]
edge_widths = [5 * (1 - G[u][v]['cost']) for u, v in G.edges()]  # inverse of cost

# Draw edges
edges = nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    edge_cmap=plt.cm.viridis_r,
    edge_vmin=0,
    edge_vmax=1,
    width=edge_widths,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=12
)

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color="skyblue")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

edge_colors = [G[u][v]['delay'] for u, v in G.edges()]
edge_widths = [5 * (1 - G[u][v]['cost']) for u, v in G.edges()]

nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    edge_color=edge_colors,
    edge_cmap=plt.cm.viridis_r,
    edge_vmin=0,
    edge_vmax=1,
    width=edge_widths,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=12
)

sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Delay (normalized)")

plt.title("Customer Journey Graph with Cost & Delay")
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt
import networkx as nx

# If not done earlier
pos = nx.spring_layout(G, seed=42)  # Stable layout

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')

# Draw all edges
nx.draw_networkx_edges(
    G, pos,
    edge_color='yellow',
    width=2,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=10
)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10)

# Draw edge labels with cost and delay
edge_labels = {
    (u, v): f"{d['cost']:.3f}, {d['delay']:.3f}"
    for u, v, d in G.edges(data=True)
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Highlight the best path
best_path = ["admin.", "cellular", "success", "yes"]
highlight_edges = list(zip(best_path[:-1], best_path[1:]))

# Draw highlighted path
nx.draw_networkx_edges(
    G, pos,
    edgelist=highlight_edges,
    edge_color='red',
    width=3,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=15
)

# Highlight path nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=best_path,
    node_color='orange',
    node_size=900
)

# Title and display
plt.title("Graph with Best Path Highlighted\n(cost, delay)")
plt.axis('off')
plt.tight_layout()
plt.show()
