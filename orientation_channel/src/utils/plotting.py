from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reliability_diagram(probs: np.ndarray, targets: np.ndarray, out_path: str, n_bins: int = 15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets)

    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_bounds[:-1] + bin_bounds[1:]) / 2.0
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i + 1])
        if mask.sum() == 0:
            bin_acc[i] = np.nan
            bin_conf[i] = np.nan
        else:
            bin_acc[i] = accuracies[mask].mean()
            bin_conf[i] = confidences[mask].mean()

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.bar(bin_centers, np.nan_to_num(bin_acc), width=1.0 / n_bins, alpha=0.6, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
