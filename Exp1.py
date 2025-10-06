from collections import deque

# --- State definitions ---
START = "EEE_WWW"
GOAL = "WWW_EEE"

def get_neighbors(state):
    """Generate all valid next states from the current state."""
    neighbors = []
    state = list(state)
    empty = state.index('_')

    # Loop through all positions to find possible moves
    for i, ch in enumerate(state):
        if ch == 'E':
            # E moves right
            if i + 1 < len(state) and state[i + 1] == '_':
                s = state.copy()
                s[i], s[i + 1] = s[i + 1], s[i]
                neighbors.append("".join(s))
            elif i + 2 < len(state) and state[i + 1] == 'W' and state[i + 2] == '_':
                s = state.copy()
                s[i], s[i + 2] = s[i + 2], s[i]
                neighbors.append("".join(s))
        elif ch == 'W':
            # W moves left
            if i - 1 >= 0 and state[i - 1] == '_':
                s = state.copy()
                s[i], s[i - 1] = s[i - 1], s[i]
                neighbors.append("".join(s))
            elif i - 2 >= 0 and state[i - 1] == 'E' and state[i - 2] == '_':
                s = state.copy()
                s[i], s[i - 2] = s[i - 2], s[i]
                neighbors.append("".join(s))
    return neighbors

# --- BFS Implementation ---
def bfs(start, goal):
    """Breadth-First Search — guarantees optimal (shortest) solution."""
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path  # Return the sequence of states
        for next_state in get_neighbors(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
    return None

# --- DFS Implementation ---
def dfs(start, goal):
    """Depth-First Search — may not give optimal solution."""
    stack = [(start, [start])]
    visited = {start}

    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        for next_state in get_neighbors(state):
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [next_state]))
    return None

# --- Run both searches ---
print("=== BFS Solution (Optimal) ===")
bfs_path = bfs(START, GOAL)
if bfs_path:
    for step in bfs_path:
        print(step)
    print(f"Total Moves: {len(bfs_path) - 1}")
else:
    print("No solution found.")

print("\n=== DFS Solution (Not necessarily optimal) ===")
dfs_path = dfs(START, GOAL)
if dfs_path:
    for step in dfs_path:
        print(step)
    print(f"Total Moves: {len(dfs_path) - 1}")
else:
    print("No solution found.")
