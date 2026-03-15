import json
import heapq
import math
import random
from collections import defaultdict

# ============================================================
# MAIN SETTINGS
# ============================================================

START_NODE = "1"
GOAL_NODE = "50"
ENERGY_BUDGET = 287932

# ============================================================
# 1.1 TASK 1 / 2 / 3
# ============================================================

def load_data_11():
    with open("G.json", "r") as f:
        G = json.load(f)

    with open("Dist.json", "r") as f:
        Dist = json.load(f)

    with open("Cost.json", "r") as f:
        Cost = json.load(f)

    with open("Coord.json", "r") as f:
        Coord = json.load(f)

    return G, Dist, Cost, Coord


# ----------------------------
# 1.1 Task 1
# Relaxed shortest path
# ----------------------------
def uniform_cost_search_relaxed(G, Dist, start, goal):
    frontier = [(0, start, [start])]
    visited = set()

    while frontier:
        total_distance, current_node, path = heapq.heappop(frontier)

        visited.add(current_node)

        if current_node == goal:
            return path, total_distance

        for neighbor in G[current_node]:
            if neighbor not in visited:
                edge_key = f"{current_node},{neighbor}"
                heapq.heappush(frontier, (total_distance + Dist[edge_key], neighbor, path + [neighbor]))

    return None, float("inf")


def compute_total_energy(path, Cost):
    total_energy = 0
    for i in range(len(path) - 1):
        edge_key = f"{path[i]},{path[i + 1]}"
        total_energy += Cost[edge_key]
    return total_energy


# ----------------------------
# 1.1 Task 2
# UCS with energy budget
# ----------------------------
def ucs_with_energy_budget(G, Dist, Cost, start, goal, budget):
    frontier = [(0, 0, start, [start])]
    best_dist = {(start, 0): 0}
    best_at_node = {start: [(0, 0)]}

    while frontier:
        total_distance, total_energy, current_node, path = heapq.heappop(frontier)

        state = (current_node, total_energy)
        if total_distance > best_dist.get(state, float("inf")):
            continue

        if current_node == goal:
            return path, total_distance, total_energy

        for neighbor in G[current_node]:
            edge_key = f"{current_node},{neighbor}"

            new_distance = total_distance + Dist[edge_key]
            new_energy = total_energy + Cost[edge_key]

            if new_energy > budget:
                continue

            dominated = False
            for old_energy, old_distance in best_at_node.get(neighbor, []):
                if old_energy <= new_energy and old_distance <= new_distance:
                    dominated = True
                    break
            if dominated:
                continue

            filtered = []
            for old_energy, old_distance in best_at_node.get(neighbor, []):
                if not (new_energy <= old_energy and new_distance <= old_distance):
                    filtered.append((old_energy, old_distance))
            filtered.append((new_energy, new_distance))
            best_at_node[neighbor] = filtered

            next_state = (neighbor, new_energy)
            if new_distance < best_dist.get(next_state, float("inf")):
                best_dist[next_state] = new_distance
                heapq.heappush(frontier, (new_distance, new_energy, neighbor, path + [neighbor]))

    return None, float("inf"), float("inf")


# ----------------------------
# 1.1 Task 3
# A* with energy budget
# ----------------------------
def heuristic(node, goal, Coord):
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruct_path_from_parent(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def a_star_with_energy_budget(G, Dist, Cost, Coord, start, goal, budget):
    frontier = []
    h_start = heuristic(start, goal, Coord)
    heapq.heappush(frontier, (h_start, 0, 0, start))

    parent = {start: None}
    best_g = {(start, 0): 0}
    best_at_node = {start: [(0, 0)]}

    while frontier:
        f, g, energy, node = heapq.heappop(frontier)

        if node == goal:
            path = reconstruct_path_from_parent(parent, goal)
            return path, g, energy

        for neighbor in G[node]:
            edge_key = f"{node},{neighbor}"

            new_g = g + Dist[edge_key]
            new_energy = energy + Cost[edge_key]

            if new_energy > budget:
                continue

            dominated = False
            if neighbor in best_at_node:
                for old_energy, old_g in best_at_node[neighbor]:
                    if old_energy <= new_energy and old_g <= new_g:
                        dominated = True
                        break
            if dominated:
                continue

            if neighbor not in best_at_node:
                best_at_node[neighbor] = []

            filtered = []
            for old_energy, old_g in best_at_node[neighbor]:
                if not (new_energy <= old_energy and new_g <= old_g):
                    filtered.append((old_energy, old_g))
            filtered.append((new_energy, new_g))
            best_at_node[neighbor] = filtered

            state = (neighbor, new_energy)
            if state not in best_g or new_g < best_g[state]:
                best_g[state] = new_g
                parent[neighbor] = node
                h = heuristic(neighbor, goal, Coord)
                heapq.heappush(frontier, (new_g + h, new_g, new_energy, neighbor))

    return None, float("inf"), float("inf")


def run_part_11():
    G, Dist, Cost, Coord = load_data_11()

    print("1.1 Task 1:")
    path, shortest_distance = uniform_cost_search_relaxed(G, Dist, START_NODE, GOAL_NODE)
    if path is None:
        print("No path found.")
    else:
        total_energy = compute_total_energy(path, Cost)
        print("Shortest path:", "->".join(path))
        print("Shortest distance:", shortest_distance)
        print("Total energy cost:", total_energy)

    print("\n1.1 Task 2:")
    path, shortest_distance, total_energy = ucs_with_energy_budget(
        G, Dist, Cost, START_NODE, GOAL_NODE, ENERGY_BUDGET
    )
    if path is None:
        print("No feasible path found.")
    else:
        print("Shortest path:", "->".join(path))
        print("Shortest distance:", shortest_distance)
        print("Total energy cost:", total_energy)

    print("\n1.1 Task 3:")
    path, shortest_distance, total_energy = a_star_with_energy_budget(
        G, Dist, Cost, Coord, START_NODE, GOAL_NODE, ENERGY_BUDGET
    )
    if path is None:
        print("No feasible path found.")
    else:
        print("Shortest path:", "->".join(path))
        print("Shortest distance:", shortest_distance)
        print("Total energy cost:", total_energy)


# ============================================================
# 2.1 TASK 1
# ============================================================

GRID_SIZE_T1 = 5
BLOCKS_T1 = {(1, 2), (3, 2)}
GOAL_STATE_T1 = (4, 4)

ACTIONS_T1 = ['U', 'D', 'L', 'R']
GAMMA_T1 = 0.9

DELTA_T1 = {
    'U': (0, 1),
    'D': (0, -1),
    'L': (-1, 0),
    'R': (1, 0),
}

LEFT_OF_T1 = {
    'U': 'L',
    'D': 'R',
    'L': 'D',
    'R': 'U',
}

RIGHT_OF_T1 = {
    'U': 'R',
    'D': 'L',
    'L': 'U',
    'R': 'D',
}


def get_states_t1():
    states = []
    for y in range(GRID_SIZE_T1):
        for x in range(GRID_SIZE_T1):
            if (x, y) not in BLOCKS_T1:
                states.append((x, y))
    return states


def move_t1(state, action):
    if state == GOAL_STATE_T1:
        return GOAL_STATE_T1

    x, y = state
    dx, dy = DELTA_T1[action]
    nx, ny = x + dx, y + dy

    if nx < 0 or nx >= GRID_SIZE_T1 or ny < 0 or ny >= GRID_SIZE_T1:
        return state
    if (nx, ny) in BLOCKS_T1:
        return state

    return (nx, ny)


def reward_t1(next_state):
    if next_state == GOAL_STATE_T1:
        return 10
    return -1


def get_transitions_t1(state, action):
    if state == GOAL_STATE_T1:
        return [(1.0, GOAL_STATE_T1, 0)]

    return [
        (0.8, move_t1(state, action), reward_t1(move_t1(state, action))),
        (0.1, move_t1(state, LEFT_OF_T1[action]), reward_t1(move_t1(state, LEFT_OF_T1[action]))),
        (0.1, move_t1(state, RIGHT_OF_T1[action]), reward_t1(move_t1(state, RIGHT_OF_T1[action])))
    ]


def compute_q_value_t1(V, state, action):
    total = 0.0
    for prob, next_state, r in get_transitions_t1(state, action):
        total += prob * (r + GAMMA_T1 * V[next_state])
    return total


def value_iteration_t1(theta=1e-10):
    states = get_states_t1()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if s == GOAL_STATE_T1:
                new_V[s] = 0.0
                continue

            best_value = max(compute_q_value_t1(V, s, a) for a in ACTIONS_T1)
            delta = max(delta, abs(best_value - V[s]))
            new_V[s] = best_value

        V = new_V

        if delta < theta:
            break

    policy = {}
    for s in states:
        if s == GOAL_STATE_T1:
            policy[s] = 'G'
        else:
            best_action = max(ACTIONS_T1, key=lambda a: compute_q_value_t1(V, s, a))
            policy[s] = best_action

    return V, policy


def policy_iteration_t1(theta=1e-10):
    states = get_states_t1()
    V = {s: 0.0 for s in states}
    policy = {}

    for s in states:
        if s == GOAL_STATE_T1:
            policy[s] = 'G'
        else:
            policy[s] = 'U'

    stable = False

    while not stable:
        while True:
            delta = 0.0
            new_V = V.copy()

            for s in states:
                if s == GOAL_STATE_T1:
                    new_V[s] = 0.0
                    continue

                action = policy[s]
                value = compute_q_value_t1(V, s, action)
                delta = max(delta, abs(value - V[s]))
                new_V[s] = value

            V = new_V
            if delta < theta:
                break

        stable = True
        for s in states:
            if s == GOAL_STATE_T1:
                continue

            old_action = policy[s]
            best_action = max(ACTIONS_T1, key=lambda a: compute_q_value_t1(V, s, a))
            policy[s] = best_action

            if best_action != old_action:
                stable = False

    return V, policy


def print_value_grid_t1(V, title):
    print(title)
    print()
    for y in reversed(range(GRID_SIZE_T1)):
        row = []
        for x in range(GRID_SIZE_T1):
            s = (x, y)
            if s in BLOCKS_T1:
                row.append("#####".rjust(8))
            else:
                row.append(f"{V[s]:.2f}".rjust(8))
        print("".join(row))
    print()


def print_policy_grid_t1(policy, title):
    print(title)
    print()
    for y in reversed(range(GRID_SIZE_T1)):
        row = []
        for x in range(GRID_SIZE_T1):
            s = (x, y)
            if s in BLOCKS_T1:
                row.append("#####".rjust(6))
            else:
                row.append(policy[s].rjust(6))
        print("".join(row))
    print()


def compare_policies_t1(p1, p2):
    same = True
    for s in get_states_t1():
        if p1[s] != p2[s]:
            same = False
            break

    if same:
        print("Both policies are the same.")
    else:
        print("Both policies are different.")


def run_part_21_task1():
    print("\n2.1 Task 1:")

    V_vi, P_vi = value_iteration_t1()
    V_pi, P_pi = policy_iteration_t1()

    print_value_grid_t1(V_vi, "Task 1 - Value Iteration: Value Function")
    print_policy_grid_t1(P_vi, "Task 1 - Value Iteration: Policy")

    print_value_grid_t1(V_pi, "Task 1 - Policy Iteration: Value Function")
    print_policy_grid_t1(P_pi, "Task 1 - Policy Iteration: Policy")

    compare_policies_t1(P_vi, P_pi)


# ============================================================
# 2.1 TASK 2
# ============================================================

GRID_SIZE_T23 = 5
START_T23 = (0, 0)
GOAL_T23 = (4, 4)
BLOCKS_T23 = {(1, 2), (3, 2)}
ACTIONS_T23 = ["U", "D", "L", "R"]

ACTION_TO_DELTA_T23 = {
    "U": (0, 1),
    "D": (0, -1),
    "L": (-1, 0),
    "R": (1, 0),
}

GAMMA_T23 = 0.9
EPSILON_T23 = 0.1
ALPHA_T23 = 0.1
STEP_REWARD_T23 = -1
GOAL_REWARD_T23 = 10

ALL_STATES_T23 = [
    (x, y)
    for x in range(GRID_SIZE_T23)
    for y in range(GRID_SIZE_T23)
    if (x, y) not in BLOCKS_T23
]


def is_valid_t23(state):
    x, y = state
    return 0 <= x < GRID_SIZE_T23 and 0 <= y < GRID_SIZE_T23 and state not in BLOCKS_T23


def is_terminal_t23(state):
    return state == GOAL_T23


def move_t23(state, action):
    if is_terminal_t23(state):
        return state

    dx, dy = ACTION_TO_DELTA_T23[action]
    next_state = (state[0] + dx, state[1] + dy)
    if not is_valid_t23(next_state):
        return state
    return next_state


def reward_t23(state, action, next_state):
    if next_state == GOAL_T23:
        return GOAL_REWARD_T23
    return STEP_REWARD_T23


def perpendicular_actions_t23(action):
    if action in ("U", "D"):
        return ["L", "R"]
    return ["U", "D"]


def sample_stochastic_step_t23(state, action):
    if is_terminal_t23(state):
        return state, 0

    side1, side2 = perpendicular_actions_t23(action)
    r = random.random()

    if r < 0.8:
        actual_action = action
    elif r < 0.9:
        actual_action = side1
    else:
        actual_action = side2

    next_state = move_t23(state, actual_action)
    return next_state, reward_t23(state, action, next_state)


def argmax_q_t23(Q, state):
    best_action = None
    best_value = float("-inf")
    for a in ACTIONS_T23:
        if Q[(state, a)] > best_value:
            best_value = Q[(state, a)]
            best_action = a
    return best_action


def epsilon_greedy_action_t23(Q, state, epsilon=EPSILON_T23):
    if random.random() < epsilon:
        return random.choice(ACTIONS_T23)
    return argmax_q_t23(Q, state)


def generate_episode_t23(Q, max_steps=200):
    episode = []
    state = START_T23

    for _ in range(max_steps):
        if is_terminal_t23(state):
            break

        action = epsilon_greedy_action_t23(Q, state)
        next_state, r = sample_stochastic_step_t23(state, action)
        episode.append((state, action, r))
        state = next_state

        if is_terminal_t23(state):
            break

    return episode


def monte_carlo_control_t23(num_episodes=10000):
    Q = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode_t23(Q)

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = GAMMA_T23 * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                returns[(s, a)].append(G)
                Q[(s, a)] = sum(returns[(s, a)]) / len(returns[(s, a)])

    policy = {}
    V = {}

    for s in ALL_STATES_T23:
        if is_terminal_t23(s):
            continue
        policy[s] = argmax_q_t23(Q, s)
        V[s] = max(Q[(s, a)] for a in ACTIONS_T23)

    V[GOAL_T23] = 0.0
    return Q, V, policy


def print_values_t23(V, title):
    print(f"\n{title}")
    for y in reversed(range(GRID_SIZE_T23)):
        row = []
        for x in range(GRID_SIZE_T23):
            s = (x, y)
            if s in BLOCKS_T23:
                row.append("#####".rjust(8))
            else:
                row.append(f"{V.get(s, 0):6.2f}".rjust(8))
        print(" ".join(row))


def print_policy_t23(policy, title):
    print(f"\n{title}")
    for y in reversed(range(GRID_SIZE_T23)):
        row = []
        for x in range(GRID_SIZE_T23):
            s = (x, y)
            if s in BLOCKS_T23:
                row.append("#####".rjust(6))
            elif s == GOAL_T23:
                row.append(" G ".rjust(6))
            else:
                row.append(policy.get(s, "?").rjust(6))
        print(" ".join(row))


def run_part_21_task2():
    print("\n2.1 Task 2:")
    Q, V, policy = monte_carlo_control_t23(num_episodes=10000)
    print_values_t23(V, "Task 2 - Monte Carlo: Value Function")
    print_policy_t23(policy, "Task 2 - Monte Carlo: Policy")


# ============================================================
# 2.1 TASK 3
# ============================================================

def q_learning_t23(num_episodes=10000, max_steps=200):
    Q = defaultdict(float)

    for _ in range(num_episodes):
        state = START_T23

        for _ in range(max_steps):
            if is_terminal_t23(state):
                break

            action = epsilon_greedy_action_t23(Q, state)
            next_state, r = sample_stochastic_step_t23(state, action)

            if is_terminal_t23(next_state):
                best_next = 0
            else:
                best_next = max(Q[(next_state, a)] for a in ACTIONS_T23)

            Q[(state, action)] += ALPHA_T23 * (
                r + GAMMA_T23 * best_next - Q[(state, action)]
            )

            state = next_state

    policy = {}
    V = {}

    for s in ALL_STATES_T23:
        if is_terminal_t23(s):
            continue
        policy[s] = argmax_q_t23(Q, s)
        V[s] = max(Q[(s, a)] for a in ACTIONS_T23)

    V[GOAL_T23] = 0.0
    return Q, V, policy


def run_part_21_task3():
    print("\n2.1 Task 3:")
    Q, V, policy = q_learning_t23(num_episodes=10000)
    print_values_t23(V, "Task 3 - Q-Learning: Value Function")
    print_policy_t23(policy, "Task 3 - Q-Learning: Policy")


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(42)

    run_part_11()
    run_part_21_task1()
    run_part_21_task2()
    run_part_21_task3()


if __name__ == "__main__":
    main()