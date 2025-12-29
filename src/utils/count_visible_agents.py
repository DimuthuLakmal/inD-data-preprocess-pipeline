import os
import pickle
from pathlib import Path

from matplotlib import pyplot as plt

filename = "data.pkl"
input_path = "../../data"
annotations_path = "../../data/annotations"

# Check data file exists
index_file_path = Path(os.path.join(input_path, filename))

data_dict = pickle.load(open(index_file_path, "rb"))
annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]

total_agents = 0
num_agent_dict = {}
for file in annotation_files:
    key = file.split('.')[0]
    record = data_dict[key]
    num_agents = len(list(record['historical_adjacent_obs'].keys()))  # Shape: (num_agents, history_length, feature_dim)
    total_agents += num_agents

    # Fill the num_agent_dict for further analysis if needed
    if num_agents not in num_agent_dict:
        num_agent_dict[num_agents] = 1
    else:
        num_agent_dict[num_agents] += 1

print(f'Total number of agents in the dataset: {total_agents}')
print('average number of agents per frame:', total_agents / len(annotation_files))

# Extract keys and values for plotting
# Use list() for compatibility across Python versions
num_agent_keys = list(num_agent_dict.keys())
num_agent_vals = list(num_agent_dict.values())

# Create the bar chart
plt.figure(figsize=(8, 6)) # Optional: adjust figure size
plt.bar(num_agent_keys, num_agent_vals, color='skyblue')

# Add labels and title
plt.xlabel('Num of Agents per Scene')
plt.ylabel('Frequency')

# Display the plot
plt.show()
