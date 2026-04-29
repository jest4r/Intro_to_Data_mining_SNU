import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# set your file location
file_loc = 'output3b.txt'
labels_loc = 'ratings_test_labels.txt'
save_path = 'roc_curve.png'

# Read the output text file
output = pd.read_csv(
    file_loc,
    sep=',',
    header=None,
    names=['user_id', 'movie_id', 'score_for_movie_id', 'timestamp']
)

# Read the true labels file
labels_df = pd.read_csv(
    labels_loc,
    sep=',',
    header=None,
    names=['user_id', 'movie_id', 'true_label', 'timestamp']
)

# Make sure row alignment is correct
if len(output) != len(labels_df):
    raise ValueError("Output and label files have different numbers of rows.")

# Optional but safer: verify (user_id, movie_id, timestamp) alignment
same_pairs = (
    (output['user_id'].values == labels_df['user_id'].values).all()
    and (output['movie_id'].values == labels_df['movie_id'].values).all()
    and (output['timestamp'].values == labels_df['timestamp'].values).all()
)
if not same_pairs:
    raise ValueError("Output rows and label rows are not aligned.")

true_labels = labels_df['true_label'].tolist()
predictions = output['score_for_movie_id'].tolist()

# ROC-AUC score
roc_auc = roc_auc_score(true_labels, predictions)
print(f"ROC-AUC Score: {roc_auc:.6f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()

# Save
plt.savefig(save_path, dpi=200)
print(f"ROC curve saved to: {save_path}")

# Optional: show
plt.show()