import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    cm = np.array([
        [798, 167,  40],
        [205, 541,  82],
        [ 80,  92, 311]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['C-0', 'C-1', 'C-2'],
                yticklabels=['C-0', 'C-1', 'C-2'],
                annot_kws={"size": 14}) # 字体放大更清晰
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix_reconstructed.png', dpi=300)


if __name__ == "__main__":
    main()
