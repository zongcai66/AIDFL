import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def process_file(file_path, exclude_clients):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    date_times, client_ids, accuracies = [], [], []

    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - client ID: (\d+) -- Accuracy: (\d+\.\d+) %"
    for line in lines:
        match = re.match(pattern, line)
        if match and int(match.group(2)) not in exclude_clients:
            date_times.append(match.group(1))
            client_ids.append(int(match.group(2)))
            accuracies.append(float(match.group(3)))

    df = pd.DataFrame({
        'DateTime': pd.to_datetime(date_times, format="%Y-%m-%d %H:%M:%S,%f"),
        'ClientID': client_ids,
        'Accuracy': accuracies
    })
    clients_per_round = 10 - len(exclude_clients)
    round_accuracies = [df.iloc[i*clients_per_round:(i+1)*clients_per_round]['Accuracy'].mean() for i in range(10)]
    return round_accuracies

scenarios = {
    "IID": ("fmnist_AIDFL_7_data_iid.txt", [1, 3, 4, 5, 7, 9, 10]),
    "non-IID": ("fmnist_AIDFL_7_data.txt", [1, 3, 4, 5, 7, 9, 10]),
    # "PNR = 10": ("cifar10_AIDFL_1_data.txt", [1]),
    # "PNR = 30": ("cifar10_AIDFL_3_data.txt", [1, 3, 5]),
    # "PNR = 50": ("fmnist_AIDFL_5_data.txt", [1, 3, 5, 7, 9]),
    # "PNR = 70": ("fmnist_AIDFL_7_data.txt", [1, 3, 4, 5, 7, 9, 10]),
}

results = {name: process_file(path, clients) for name, (path, clients) in scenarios.items()}

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
plt.legend(loc="lower right", prop={'size':16})
plt.grid(True)
plt.ylim(0, 100)
plt.show()
