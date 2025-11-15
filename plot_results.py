import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# Read results
df = pd.read_csv("E:/LiDA/summary_results.csv")

# --- Plot 1: Accuracy/F1 by Augmentation (English) ---
en = df[df["Dataset"]=="English"]

ax = en.plot(x="Augmentation", y=["Accuracy", "F1"], kind="bar", rot=0)
ax.set_title("English Dataset: LiDA Augmentation Comparison")
ax.set_ylim(0.75, 0.90)
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("E:/LiDA/saved/english_augmentations.png", dpi=300)
plt.close()

# --- Plot 2: Cross‑lingual Accuracy ---
cross = df[df["Augmentation"]=="All"]
cross.plot(x="Dataset", y="Accuracy", kind="bar", color="teal", legend=False)
plt.title("Cross‑lingual Accuracy (All Augmentations)")
plt.ylabel("Accuracy")
plt.ylim(0.85, 0.91)
plt.tight_layout()
plt.savefig("E:/LiDA/saved/crosslingual_accuracy.png", dpi=300)
plt.close()

# --- Plot 3: MCC Values Across Datasets ---
df.plot(x="Dataset", y="MCC", kind="bar", color="orange", legend=False)
plt.title("MCC Across Datasets/Augmentations")
plt.ylabel("MCC")
plt.tight_layout()
plt.savefig("E:/LiDA/saved/mcc_comparison.png", dpi=300)
plt.close()

print("✅ Plots saved inside E:/LiDA/saved/")
