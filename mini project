"""
Optimal Path Detection for a Vehicle
Finds the best route between a source and a destination so that
travel time and travel cost are minimised (Dijkstra and A* search).
"""
import heapq
import math
 
# ---------------- 1. ROAD NETWORK (graph) ----------------
# Node coordinates (x, y) in km - used by the A* heuristic
COORDS = {
    "S": (0, 4), "A": (3, 7), "B": (3, 4), "C": (3, 1),
    "D": (6, 7), "E": (6, 4), "F": (6, 1), "G": (9, 5), "T": (12, 4),
}
 
# (node1, node2, distance_km, time_min, toll_Rs)
ROADS = [
    ("S", "A", 5.0, 9, 0),  ("S", "B", 3.2, 7, 40), ("S", "C", 5.0, 12, 0),
    ("A", "D", 3.0, 6, 0),  ("A", "B", 3.0, 8, 0),  ("B", "E", 3.0, 5, 40),
    ("B", "C", 3.0, 7, 0),  ("C", "F", 3.0, 8, 0),  ("D", "G", 4.5, 8, 0),
    ("E", "G", 3.2, 6, 30), ("E", "F", 3.0, 7, 0),  ("F", "G", 5.0, 12, 0),
    ("D", "E", 3.0, 6, 0),  ("G", "T", 3.2, 6, 0),  ("E", "T", 6.0, 8, 60),
    ("F", "T", 6.7, 15, 0),
]
 
FUEL_COST_PER_KM = 6.5      # Rs per km
TIME_VALUE_PER_MIN = 5.0    # Rs value of one minute of travel time
 
 
def build_graph():
    graph = {n: [] for n in COORDS}
    for u, v, d, t, toll in ROADS:
        cost = d * FUEL_COST_PER_KM + toll
        graph[u].append((v, d, t, cost))
        graph[v].append((u, d, t, cost))
    return graph
 
 
# ---------------- 2. EDGE WEIGHT BY CRITERION ----------------
def weight(edge, criterion):
    _, d, t, c = edge
    if criterion == "time":
        return t
    if criterion == "cost":
        return c
    return c + TIME_VALUE_PER_MIN * t        # "balanced"
 
 
def heuristic(n, goal, criterion):
    (x1, y1), (x2, y2) = COORDS[n], COORDS[goal]
    dist = math.hypot(x1 - x2, y1 - y2)      # never over-estimates
    if criterion == "time":
        return dist * 1.0                    # fastest possible ~1 min/km (admissible)
    if criterion == "cost":
        return dist * FUEL_COST_PER_KM
    return dist * FUEL_COST_PER_KM + TIME_VALUE_PER_MIN * dist
 
 
# ---------------- 3. SEARCH ALGORITHMS ----------------
def shortest_path(graph, start, goal, criterion="time", use_astar=False):
    pq = [(0, 0, start, [start])]
    best = {start: 0}
    expanded = 0
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if g > best.get(node, math.inf):
            continue
        expanded += 1
        if node == goal:
            return path, g, expanded
        for edge in graph[node]:
            nxt = edge[0]
            ng = g + weight(edge, criterion)
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                h = heuristic(nxt, goal, criterion) if use_astar else 0
                heapq.heappush(pq, (ng + h, ng, nxt, path + [nxt]))
    return None, math.inf, expanded
 
 
# ---------------- 4. ROUTE SUMMARY ----------------
def summarise(graph, path):
    dist = time = cost = 0
    for a, b in zip(path, path[1:]):
        _, d, t, c = next(e for e in graph[a] if e[0] == b)
        dist += d; time += t; cost += c
    return dist, time, cost
 
 
# ---------------- 5. VISUALISATION ----------------
def plot_routes(graph, routes, filename="optimal_path.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for u, v, d, t, toll in ROADS:                      # all roads in grey
        (x1, y1), (x2, y2) = COORDS[u], COORDS[v]
        ax.plot([x1, x2], [y1, y2], color="lightgrey", lw=2, zorder=1)
    styles = {"time": ("red", "Fastest (time)"),
              "cost": ("green", "Cheapest (cost)"),
              "balanced": ("blue", "Balanced")}
    for crit, path in routes.items():                   # highlighted routes
        colour, label = styles[crit]
        xs = [COORDS[n][0] for n in path]
        ys = [COORDS[n][1] + {"time": .12, "cost": -.12, "balanced": 0}[crit] for n in path]
        ax.plot(xs, ys, color=colour, lw=3, label=label, zorder=2)
    for n, (x, y) in COORDS.items():
        ax.scatter(x, y, s=500, color="white", edgecolor="black", zorder=3)
        ax.text(x, y, n, ha="center", va="center", fontweight="bold", zorder=4)
    ax.set_title("AI Project - Optimal Path (S = Source, T = Destination)")
    ax.legend(loc="upper left"); ax.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
 
 
if __name__ == "__main__":
    g = build_graph()
    src, dst = "S", "T"
    print("OPTIMAL PATH DETECTION FOR A VEHICLE")
    print("=" * 62)
    routes = {}
    for crit in ("time", "cost", "balanced"):
        for name, astar in (("Dijkstra", False), ("A*", True)):
            path, _, exp = shortest_path(g, src, dst, crit, astar)
            d, t, c = summarise(g, path)
            routes[crit] = path
            print(f"[{crit.upper():8}] {name:8} -> {' -> '.join(path)}")
            print(f"           Distance: {d:5.1f} km | Time: {t:3d} min | "
                  f"Cost: Rs {c:6.1f} | Nodes expanded: {exp}")
        print("-" * 62)
    plot_routes(g, routes)
    print("Route map saved as optimal_path.png")
