import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# load both prediction outputs
linear = pd.read_csv("E:/LiDA/saved/test_predictions.csv")
ae = pd.read_csv("E:/LiDA/saved/test_predictions_ae.csv")

print("=== Linear Augmentation ===")
print("Accuracy:", accuracy_score(linear.true_label, linear.predicted_label))
print("F1 Score:", f1_score(linear.true_label, linear.predicted_label))
print("MCC:", matthews_corrcoef(linear.true_label, linear.predicted_label))

print("\n=== AutoEncoder Augmentation ===")
print("Accuracy:", accuracy_score(ae.true_label, ae.predicted_label))
print("F1 Score:", f1_score(ae.true_label, ae.predicted_label))
print("MCC:", matthews_corrcoef(ae.true_label, ae.predicted_label))
