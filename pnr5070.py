import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


# Function to process file and calculate round accuracies
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
    round_accuracies = [
        df.iloc[i * clients_per_round:(i + 1) * clients_per_round][
            'Accuracy'].mean() for i in range(10)]
    return round_accuracies


# Define scenarios for MNIST and FashionMNIST
mnist_scenarios = {
    "PNR = 50": {
        "IID": ("mnist_AIDFL_5_label_iid.txt", [1, 3, 5, 7, 9]),
        "non-IID": ("mnist_AIDFL_5_label.txt", [1, 3, 5, 7, 9])
    },
    "PNR = 70": {
        "IID": ("mnist_AIDFL_7_label_iid.txt", [1, 3, 4, 5, 7, 9, 10]),
        "non-IID": ("mnist_AIDFL_7_label.txt", [1, 3, 4, 5, 7, 9, 10])
    }
}

fmnist_scenarios = {
    "PNR = 50": {
        "IID": ("fmnist_AIDFL_5_label_iid.txt", [1, 3, 5, 7, 9]),
        "non-IID": ("fmnist_AIDFL_5_label.txt", [1, 3, 5, 7, 9])
    },
    "PNR = 70": {
        "IID": ("fmnist_AIDFL_7_label_iid.txt", [1, 3, 4, 5, 7, 9, 10]),
        "non-IID": ("fmnist_AIDFL_7_label.txt", [1, 3, 4, 5, 7, 9, 10])
    }
}

# Plotting the results in a 2x2 layout
rounds = list(range(1, 11))
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

# Titles for different poisoning scenarios
poisoned_titles = ["PNR = 50", "PNR = 70"]

for idx, poison_label in enumerate(poisoned_titles):
    # MNIST plot
    mnist_conditions = mnist_scenarios[poison_label]
    ax_mnist = axs[0, idx]
    for condition, (file_path, exclude_clients) in mnist_conditions.items():
        accuracies = process_file(file_path, exclude_clients)
        linestyle = '-' if condition == "IID" else '--'
        ax_mnist.plot(rounds, accuracies, marker='o', linestyle=linestyle,
                      label=condition)
    ax_mnist.set_title(f"MNIST ({poison_label})", fontsize=18)
    ax_mnist.set_xlabel("Round", fontsize=18)
    ax_mnist.set_ylabel("Accuracy (%)", fontsize=18)
    ax_mnist.set_yticks(np.arange(0, 101, 10))
    ax_mnist.tick_params(axis='both', labelsize=18)
    ax_mnist.grid(True)

    # FashionMNIST plot
    fmnist_conditions = fmnist_scenarios[poison_label]
    ax_fmnist = axs[1, idx]
    for condition, (file_path, exclude_clients) in fmnist_conditions.items():
        accuracies = process_file(file_path, exclude_clients)
        linestyle = '-' if condition == "IID" else '--'
        ax_fmnist.plot(rounds, accuracies, marker='o', linestyle=linestyle,
                       label=condition)
    ax_fmnist.set_title(f"FashionMNIST ({poison_label})", fontsize=18)
    ax_fmnist.set_xlabel("Round", fontsize=18)
    ax_fmnist.tick_params(axis='both', labelsize=18)
    ax_fmnist.grid(True)

# Add legend
fig.legend(["IID", "non-IID"], loc="upper center", ncol=2, fontsize=18,
           bbox_to_anchor=(0.5, 1))

plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout for the legend
plt.show()
