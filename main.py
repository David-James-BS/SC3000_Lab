import json
import heapq
import math
import random
import time
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
        frontier.sort()
        total_distance, current_node, path = frontier.pop(0)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            return path, total_distance

        for neighbor in G[current_node]:
            if neighbor not in visited:
                edge_key = f"{current_node},{neighbor}"
                frontier.append(
                    (total_distance + Dist[edge_key], neighbor, path + [neighbor])
                )

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
    visited = {}

    while frontier:
        frontier.sort()
        total_distance, total_energy, current_node, path = frontier.pop(0)

        if current_node in visited:
            prev_dist, prev_energy = visited[current_node]
            if total_distance >= prev_dist and total_energy >= prev_energy:
                continue

        visited[current_node] = (total_distance, total_energy)

        if current_node == goal:
            return path, total_distance, total_energy

        for neighbor in G[current_node]:
            edge_key = f"{current_node},{neighbor}"

            new_distance = total_distance + Dist[edge_key]
            new_energy = total_energy + Cost[edge_key]

            if new_energy <= budget:
                frontier.append(
                    (new_distance, new_energy, neighbor, path + [neighbor])
                )

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

def compare_policies_detailed(policy_a, policy_b, label_a, label_b, states, goal_state):
    matching_states = 0
    differing_states = []

    for state in states:
        if state == goal_state:
            continue

        action_a = policy_a.get(state, "?")
        action_b = policy_b.get(state, "?")

        if action_a == action_b:
            matching_states += 1
        else:
            differing_states.append((state, action_a, action_b))

    total_states = len(states) - 1  # exclude goal
    similarity = (matching_states / total_states) * 100 if total_states > 0 else 0.0

    print(f"Comparison: {label_a} vs {label_b}")
    print(f"Matching states: {matching_states}/{total_states}")
    print(f"Different states: {len(differing_states)}/{total_states}")
    print(f"Similarity: {similarity:.2f}%")

    if len(differing_states) == 0:
        print("Policies are identical.")
    else:
        print("Differing states:")
        for state, action_a, action_b in differing_states:
            print(f"  State {state}: {label_a}={action_a}, {label_b}={action_b}")
    print()


def print_value_difference_grid(V_a, V_b, title, grid_size, blocks):
    print(title)
    print()

    for y in reversed(range(grid_size)):
        row = []
        for x in range(grid_size):
            state = (x, y)
            if state in blocks:
                row.append("#####".rjust(8))
            else:
                diff = abs(V_a.get(state, 0.0) - V_b.get(state, 0.0))
                row.append(f"{diff:.2f}".rjust(8))
        print("".join(row))
    print()


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

    return V_vi, P_vi


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
    print(title)
    print()

    for y in reversed(range(GRID_SIZE_T23)):
        row = []
        for x in range(GRID_SIZE_T23):
            s = (x, y)

            if s in BLOCKS_T23:
                row.append("#####".rjust(8))
            else:
                row.append(f"{V.get(s, 0.0):.2f}".rjust(8))

        print("".join(row))

    print()


def print_policy_t23(policy, title):
    print(title)
    print()

    for y in reversed(range(GRID_SIZE_T23)):
        row = []
        for x in range(GRID_SIZE_T23):
            s = (x, y)

            if s in BLOCKS_T23:
                row.append("#####".rjust(6))
            elif s == GOAL_T23:
                row.append("G".rjust(6))
            else:
                row.append(policy.get(s, "?").rjust(6))

        print("".join(row))

    print()


def run_part_21_task2():
    print("\n2.1 Task 2:\n")

    Q_mc, V_mc, P_mc = monte_carlo_control_t23(num_episodes=10000)

    print_values_t23(V_mc, "Task 2 - Monte Carlo: Value Function")
    print_policy_t23(P_mc, "Task 2 - Monte Carlo: Policy")

    return V_mc, P_mc


# ============================================================
# 2.1 TASK 3
# ============================================================

def q_learning_t23(num_episodes=10000, max_steps=200, optimal_policy=None, window=100):
    Q = defaultdict(float)

    episode_returns = []
    episode_lengths = []
    episode_success = []
    policy_match_episode = None

    start_time = time.perf_counter()

    for episode_index in range(1, num_episodes + 1):
        state = START_T23
        total_return = 0
        steps = 0
        reached_goal = False

        for _ in range(max_steps):
            if is_terminal_t23(state):
                reached_goal = True
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
            total_return += r
            steps += 1

            if is_terminal_t23(state):
                reached_goal = True
                break

        episode_returns.append(total_return)
        episode_lengths.append(steps)
        episode_success.append(1 if reached_goal else 0)

        if optimal_policy is not None and policy_match_episode is None:
            current_policy = {}
            for s in ALL_STATES_T23:
                if is_terminal_t23(s):
                    continue
                current_policy[s] = argmax_q_t23(Q, s)

            same = True
            for s in ALL_STATES_T23:
                if s == GOAL_T23:
                    continue
                if current_policy.get(s, "?") != optimal_policy.get(s, "?"):
                    same = False
                    break

            if same:
                policy_match_episode = episode_index

    end_time = time.perf_counter()

    policy = {}
    V = {}

    for s in ALL_STATES_T23:
        if is_terminal_t23(s):
            continue
        policy[s] = argmax_q_t23(Q, s)
        V[s] = max(Q[(s, a)] for a in ACTIONS_T23)

    V[GOAL_T23] = 0.0

    stats = {
        "training_time_seconds": end_time - start_time,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_success": episode_success,
        "policy_match_episode": policy_match_episode,
        "window": window,
    }

    return Q, V, policy, stats

def run_part_21_task3(optimal_policy):
    print("\n2.1 Task 3:\n")

    Q, V_ql, P_ql, stats = q_learning_t23(
        num_episodes=10000,
        optimal_policy=optimal_policy,
        window=100
    )

    print_values_t23(V_ql, "Task 3 - Q-Learning: Value Function")
    print_policy_t23(P_ql, "Task 3 - Q-Learning: Policy")

    print_learning_statistics(
        stats,
        "Q-Learning Convergence / Efficiency Statistics:"
    )

    return V_ql, P_ql

def print_learning_statistics(stats, title):
    print(title)

    returns = stats["episode_returns"]
    lengths = stats["episode_lengths"]
    success = stats["episode_success"]
    window = stats["window"]

    last_returns = returns[-window:] if len(returns) >= window else returns
    last_lengths = lengths[-window:] if len(lengths) >= window else lengths
    last_success = success[-window:] if len(success) >= window else success

    avg_return = sum(last_returns) / len(last_returns) if last_returns else 0.0
    avg_length = sum(last_lengths) / len(last_lengths) if last_lengths else 0.0
    success_rate = (sum(last_success) / len(last_success) * 100) if last_success else 0.0

    print(f"Training time: {stats['training_time_seconds']:.4f} seconds")
    print(f"Average return over last {len(last_returns)} episodes: {avg_return:.2f}")
    print(f"Average episode length over last {len(last_lengths)} episodes: {avg_length:.2f}")
    print(f"Success rate over last {len(last_success)} episodes: {success_rate:.2f}%")

    if stats["policy_match_episode"] is None:
        print("First full match with optimal Task 1 policy: Not reached")
    else:
        print(f"First full match with optimal Task 1 policy: Episode {stats['policy_match_episode']}")
    print()

def run_part_21_comparisons(P_opt, P_mc, P_ql):
    print("\n2.1 Policy Comparisons:\n")

    print("Task 2 vs Task 1 Optimal Policy:")
    compare_policies_detailed(
        P_mc,
        P_opt,
        "Monte Carlo",
        "Task 1 Optimal",
        ALL_STATES_T23,
        GOAL_T23
    )

    print("Task 3 vs Task 2:")
    compare_policies_detailed(
        P_ql,
        P_mc,
        "Q-Learning",
        "Monte Carlo",
        ALL_STATES_T23,
        GOAL_T23
    )

    print("Task 3 vs Task 1 Optimal Policy:")
    compare_policies_detailed(
        P_ql,
        P_opt,
        "Q-Learning",
        "Task 1 Optimal",
        ALL_STATES_T23,
        GOAL_T23
    )
# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(42)

    run_part_11()

    # Task 1
    V_opt, P_opt = run_part_21_task1()

    # Task 2
    V_mc, P_mc = run_part_21_task2()

    # Task 3
    V_ql, P_ql = run_part_21_task3(P_opt)

    # Comparisons
    run_part_21_comparisons(P_opt, P_mc, P_ql)


if __name__ == "__main__":
    main()