import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure directory exists
os.makedirs('c:/DetectionGuard/research_outputs', exist_ok=True)

results = {
    'malicious_ratio': [0.0, 0.3],
    'Federated Averaging': [0.9648, 0.8982],
    'DetectionGuard (Robust)': [0.9636, 0.9311]
}

# Print Table
print("| % Malicious | FedAvg Accuracy | DetectionGuard Accuracy |")
print("|---|---|---|")
for i, ratio in enumerate(results['malicious_ratio']):
    print(f"| {int(ratio*100)}% | {results['Federated Averaging'][i]:.4f} | {results['DetectionGuard (Robust)'][i]:.4f} |")

# Save Chart
labels = [f'{p*100:.0f}%' for p in results['malicious_ratio']]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, results['Federated Averaging'], width, label='Federated Averaging')
rects2 = ax.bar(x + width/2, results['DetectionGuard (Robust)'], width, label='DetectionGuard (Robust)')

ax.set_ylabel('Global Model Accuracy')
ax.set_xlabel('Percentage of Malicious Clients')
ax.set_title('Model Accuracy vs. Percentage of Malicious Clients')
ax.set_xticks(x, labels)
ax.legend()
ax.set_ylim(0, 1.1)
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

fig.tight_layout()
plt.savefig('c:/DetectionGuard/research_outputs/research_comparison_chart.png')
print("Chart saved.")
