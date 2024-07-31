import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Define a function to process a file and return average accuracies
def process_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store the extracted data
    date_times = []
    client_ids = []
    accuracies = []

    # Regex pattern to parse the log entries
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - client ID: (\d+) -- Accuracy: (\d+\.\d+) %"

    # Process each line in the log file
    for line in lines:
        match = re.match(pattern, line)
        if match:
            date_times.append(match.group(1))
            client_ids.append(int(match.group(2)))
            accuracies.append(float(match.group(3)))

    # Create a DataFrame from the parsed data
    df = pd.DataFrame({
        'DateTime': pd.to_datetime(date_times, format="%Y-%m-%d %H:%M:%S,%f"),
        'ClientID': client_ids,
        'Accuracy': accuracies
    })

    # Exclude data from client ID 1
    df_filtered = df[df['ClientID'] != 1]

    # Calculate average accuracy for each round (assuming 10 rounds)
    round_accuracies = [df_filtered.iloc[i*9:(i+1)*9]['Accuracy'].mean() for i in range(10)]
    return round_accuracies

# File paths
files = {
    "AIDFL": "fmnist_AIDFL_1_label.txt",
    "AIDFL + Krum": "fmnist_Krum_1_label.txt",
    "AIDFL + Median": "fmnist_Median_1_label.txt",
    "AIDFL + TrimmedMean": "fmnist_TrimmedMean_1_label.txt"
}

# Process each file
results = {name: process_file(path) for name, path in files.items()}

# Plotting the results
rounds = list(range(1, 11))
plt.figure(figsize=(10, 8))
for name, accuracies in results.items():
    plt.plot(rounds, accuracies, marker='o', linestyle='-', label=name)
    sem_accuracy = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    print(accuracies)
    print(sem_accuracy)

plt.xlabel('Round', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.xticks(rounds, fontsize=16)
plt.yticks(fontsize=14)# Label all rounds
plt.grid(True)
plt.legend(loc="lower right", prop = {'size':16})
# plt.tight_layout()
plt.ylim(0, 100)
plt.show()
