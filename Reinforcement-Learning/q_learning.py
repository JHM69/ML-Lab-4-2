import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


# Scenario Context: Journey of Mohammadpur ↔ Jagannath University (Sadarghat)

# I travel between my home in Mohammadpur and JnU in Sadarghat under varying conditions:
# - Transportation Modes:
#   - Dhurjoy (University) Bus: Free, but only available early morning to JnU (~1 hr) and returns at 3:30 PM (~1.5 hrs).
#   - Metro Rail Route (if missing Dhurjoy or returning early): Bus to Farmgate (10 TK, 15 mins) → Metro to Motijheel (30 TK, 10 mins) → Rickshaw to Sadarghat (50 TK, 25 mins). Total ~90 TK, ~50 mins.
#   - Local Buses: Cheaper (~30 TK total) but can be slow and affected by traffic, especially around Gulistan.
#   - Uber Bike: Fastest option if urgent and you have enough money (≥200 TK).

# - Conditions Affecting Decisions:
#   - Time Scenario: Early (Dhurjoy Bus), Late (Metro/Uber if urgent), Return trips (Dhurjoy after 3:30 PM, Metro or Uber if early and urgent).
#   - Traffic Levels: High traffic favors Metro or Uber to save time.
#   - Money Levels: Low funds push towards Dhurjoy or Local Bus; sufficient funds allow Metro or Uber.
#   - Urgency: Higher urgency increases preference for faster but more expensive options (Uber Bike or Metro).

# The code uses Q-learning to learn the best action (transport mode) for each combination of time, traffic, money, and urgency, aiming to minimize travel time and cost (negative rewards) and choose the most efficient means of travel.

# This project applies Q-learning to select transportation options under various conditions. The agent learns an optimal policy that balances **time**, **cost**, and **urgency**.

# -----------------------------
# Constants for Scenarios, Traffic, Money, Urgency, and Actions
# -----------------------------
GOING_EARLY = 0
GOING_LATE = 1
RETURN_BEFORE_2PM = 2
RETURN_AFTER_2PM = 3
time_scenarios = [GOING_EARLY, GOING_LATE, RETURN_BEFORE_2PM, RETURN_AFTER_2PM]  # 0=GoingEarly, 1=GoingLate, 2=ReturnBefore2PM, 3=ReturnAfter2PM

LOW_TRAFFIC = 0
MEDIUM_TRAFFIC = 1
HIGH_TRAFFIC = 2
traffic_levels = [LOW_TRAFFIC, MEDIUM_TRAFFIC, HIGH_TRAFFIC]  # 0=Low, 1=Medium, 2=High

LOW_MONEY = 0
MEDIUM_MONEY = 1
HIGH_MONEY = 2
money_levels = [LOW_MONEY, MEDIUM_MONEY, HIGH_MONEY]  # 0:<90, 1:90-199, 2:≥200

NOT_URGENT = 0
MEDIUM_URGENCY = 1
VERY_URGENT = 2
urgency_levels = [NOT_URGENT, MEDIUM_URGENCY, VERY_URGENT]  # 0=Not Urgent, 1=Medium Urgency, 2=Very Urgent

DHURJOY_BUS = 0
METRO = 1
LOCAL_BUS = 2
UBER_BIKE = 3
actions = [DHURJOY_BUS, METRO, LOCAL_BUS, UBER_BIKE]  # 0:Dhurjoy Bus, 1=Metro, 2=Local Bus, 3=Uber Bike

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.05
num_episodes = 30000

# Base factors for reward
A_time = 0.1
A_cost = 0.01

def scenario_name(ts):
    return ["GoingEarly", "GoingLate", "ReturnBefore2PM", "ReturnAfter2PM"][ts]

def traffic_name(tf):
    return ["Low", "Medium", "High"][tf]

def money_name(m):
    if m == 0:
        return "<90TK"
    elif m == 1:
        return "90-199TK"
    else:
        return "200TK or more"

def urgency_name(u):
    return ["Not Urgent", "Medium Urgency", "Very Urgent"][u]

def action_name(a):
    return ["Dhurjoy Bus", "Metro", "Local Bus", "Uber Bike"][a]

def state_to_index(ts, tf, m, u):
    # Now we have 4 dimensions: time scenario, traffic, money, urgency
    # total states = 4 * 3 * 3 * 3 = 108
    return ts * 27 + tf * 9 + m * 3 + u

def insufficient_funds_penalty():
    return (50, 1000)

def get_time_cost(action, ts, tf, m):
    # Same logic as before, unchanged except we don't handle urgency here
    if ts == 0:
        dhurjoy_time, dhurjoy_cost = 30, 0
        metro_time, metro_cost = 50, 90
        local_time, local_cost = 60, 30
        uber_time, uber_cost = 25, 200
        dhurjoy_available = True
    elif ts == 1:
        dhurjoy_available = False
        base_metro_time, base_metro_cost = 50, 90
        base_local_time = 60 + (tf * 10)
        if tf == 2: base_local_time = 90
        base_local_cost = 30
        if tf == 0:
            uber_time, uber_cost = 20, 200
        elif tf == 1:
            uber_time, uber_cost = 25, 200
        else:
            uber_time, uber_cost = 30, 200
        dhurjoy_time, dhurjoy_cost = 120, 0
        metro_time, metro_cost = base_metro_time, base_metro_cost
        local_time, local_cost = base_local_time, base_local_cost
    elif ts == 2:
        dhurjoy_available = False
        base_metro_time, base_metro_cost = 50, 90
        base_local_time = 60 + (tf * 10)
        if tf == 2: base_local_time = 90
        base_local_cost = 30
        if tf == 0:
            uber_time, uber_cost = 25, 200
        elif tf == 1:
            uber_time, uber_cost = 30, 200
        else:
            uber_time, uber_cost = 35, 200
        dhurjoy_time, dhurjoy_cost = 120, 0
        metro_time, metro_cost = base_metro_time, base_metro_cost
        local_time, local_cost = base_local_time, base_local_cost
    else:
        dhurjoy_available = True
        dhurjoy_time, dhurjoy_cost = 90, 0
        metro_time, metro_cost = 50, 90
        local_time = 60 + (tf * 10)
        if tf == 2: local_time = 90
        local_cost = 30
        uber_time, uber_cost = 45, 200

    # Money constraints
    if action == UBER_BIKE and m < HIGH_MONEY:
        t_pen, c_pen = insufficient_funds_penalty()
        uber_time += t_pen
        uber_cost += c_pen

    if action == METRO and m < MEDIUM_MONEY:
        t_pen, c_pen = insufficient_funds_penalty()
        metro_time += t_pen
        metro_cost += c_pen

    # If m=0 and Dhurjoy available, choosing non-Dhurjoy = penalty
    if m == LOW_MONEY and dhurjoy_available and action != DHURJOY_BUS:
        if action == METRO:
            metro_time += 100; metro_cost += 500
        elif action == LOCAL_BUS:
            local_time += 100; local_cost += 500
        elif action == UBER_BIKE:
            uber_time += 100; uber_cost += 500

    # If Dhurjoy chosen but not available
    if action == DHURJOY_BUS and not dhurjoy_available:
        dhurjoy_time += 100
        dhurjoy_cost += 500

    # Return chosen action time/cost
    if action == DHURJOY_BUS:
        return dhurjoy_time, dhurjoy_cost
    elif action == METRO:
        return metro_time, metro_cost
    elif action == LOCAL_BUS:
        return local_time, local_cost
    else:
        return uber_time, uber_cost

def get_reward(time, cost, urgency):
    # We scale the time penalty based on urgency.
    # If urgency=2 (Very Urgent), we penalize time more heavily.
    # If urgency=0 (Not Urgent), we penalize time less heavily.
    # Example: A_time factor is multiplied by (1 + urgency), so:
    # urgency=0 -> A_time * 1
    # urgency=1 -> A_time * 2
    # urgency=2 -> A_time * 3
    time_factor = A_time * (1 + urgency)
    return - (time_factor * time + A_cost * cost)

# Initialize Q with the new dimensionality
Q = np.zeros((4 * 3 * 3 * 3, 4))
rewards_history = []
actions_chosen = []
epsilon_values = []

for ep in range(num_episodes):
    ts = np.random.choice(time_scenarios)
    tf = np.random.choice(traffic_levels)
    m = np.random.choice(money_levels)
    u = np.random.choice(urgency_levels)

    s_idx = state_to_index(ts, tf, m, u)

    if np.random.rand() < epsilon:
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q[s_idx])

    time, cost = get_time_cost(action, ts, tf, m)
    r = get_reward(time, cost, u)

    # Q-Learning update
    # Note: In this simplified version, we don't have multiple steps per episode,
    # just one step. You can modify the environment for multi-step if needed.
    Q[s_idx, action] = Q[s_idx, action] + alpha * (r - Q[s_idx, action])

    rewards_history.append(r)
    actions_chosen.append(action)

    epsilon_values.append(epsilon)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# -----------------------------
# Visualizations of Training
# -----------------------------
window = 500
if len(rewards_history) > window:
    rolling_avg = [np.mean(rewards_history[i:i + window]) for i in range(len(rewards_history) - window)]
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg)
    plt.title("Average Reward over Time (Rolling 500 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True)
    plt.show()

# Q-table Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(Q, annot=False, cmap='viridis')
plt.title("Q-Table Heatmap")
plt.xlabel("Actions (0:Dhurjoy, 1:Metro, 2:Local, 3:Uber)")
plt.ylabel("States")
plt.show()

# Distribution of actions chosen during training
plt.figure(figsize=(6, 4))
plt.hist(actions_chosen, bins=np.arange(-0.5, 4, 1), rwidth=0.8)
plt.xticks([0, 1, 2, 3], ["Dhurjoy", "Metro", "Local", "Uber"])
plt.title("Distribution of Actions Chosen During Training")
plt.xlabel("Action")
plt.ylabel("Frequency")
plt.show()

# Epsilon decay
plt.figure(figsize=(8, 4))
plt.plot(epsilon_values)
plt.title("Epsilon Decay over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)
plt.show()

# -----------------------------
# Policy Visualization
# -----------------------------
print("Learned Policy:")
policy_data = []
best_actions_count = [0, 0, 0, 0]
for ts_val in time_scenarios:
    for tf_val in traffic_levels:
        for m_val in money_levels:
            for u_val in urgency_levels:
                s_idx = state_to_index(ts_val, tf_val, m_val, u_val)
                best_action = np.argmax(Q[s_idx])
                best_actions_count[best_action] += 1
                policy_data.append([scenario_name(ts_val), traffic_name(tf_val), money_name(m_val), urgency_name(u_val), action_name(best_action)])

print(tabulate(policy_data, headers=["Scenario", "Traffic", "Money", "Urgency", "Best Action"], tablefmt="github"))

# Pie chart of best actions across all states
plt.figure(figsize=(6, 6))
plt.pie(best_actions_count, labels=["Dhurjoy", "Metro", "Local", "Uber"], autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Best Actions Across All States")
plt.show()

# -----------------------------
# Testing
# -----------------------------

def test_scenario(ts, tf, m, u):
    s_idx = state_to_index(ts, tf, m, u)
    best_action = np.argmax(Q[s_idx])
    time, cost = get_time_cost(best_action, ts, tf, m)
    r = get_reward(time, cost, u)
    print(f"Test Scenario: {scenario_name(ts)}, Traffic={traffic_name(tf)}, Money={money_name(m)}, Urgency={urgency_name(u)}")
    print(f"Chosen Action: {action_name(best_action)}, Time={time}, Cost={cost}, Reward={r:.2f}")
    print()

# Test a few scenarios explicitly using constants
test_scenario(GOING_EARLY, LOW_TRAFFIC, LOW_MONEY, NOT_URGENT)  # GoingEarly, Low, <90TK, Not Urgent
test_scenario(GOING_LATE, HIGH_TRAFFIC, HIGH_MONEY, VERY_URGENT)  # GoingLate, High, ≥200TK, Very Urgent
test_scenario(RETURN_AFTER_2PM, LOW_TRAFFIC, LOW_MONEY, MEDIUM_URGENCY)  # ReturnAfter2PM, Low, <90TK, Medium Urgency
test_scenario(RETURN_BEFORE_2PM, MEDIUM_TRAFFIC, MEDIUM_MONEY, VERY_URGENT)  # ReturnBefore2PM, Medium, 90-199TK, Very Urgent

# Batch test all states and compute average reward
all_rewards = []
for ts_val in time_scenarios:
    for tf_val in traffic_levels:
        for m_val in money_levels:
            for u_val in urgency_levels:
                s_idx = state_to_index(ts_val, tf_val, m_val, u_val)
                best_action = np.argmax(Q[s_idx])
                time, cost = get_time_cost(best_action, ts_val, tf_val, m_val)
                r = get_reward(time, cost, u_val)
                all_rewards.append(r)

print("Average Reward for Best Actions Across All States:", np.mean(all_rewards))
