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

    # Exclude data from client IDs 1, 3, and 5
    df_filtered = df[~df['ClientID'].isin([1, 3, 4, 5, 7, 9, 10])]

    # Calculate average accuracy for each round (assuming 10 rounds)
    round_accuracies = [df_filtered.iloc[i*3:(i+1)*3]['Accuracy'].mean() for i in range(10)]
    return round_accuracies

# File paths
files = {
    "AIDFL": "mnist_AIDFL_7_data.txt",
    "AIDFL + Krum": "mnist_Krum_7_data.txt",
    "AIDFL + Median": "mnist_Median_7_data.txt",
    "AIDFL + TrimmedMean": "mnist_TrimmedMean_7_data.txt"
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
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(loc="lower right", prop={'size':16})
plt.ylim(0, 100)
plt.show()
