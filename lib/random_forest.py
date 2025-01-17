import numpy as np
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import *

def evaluate_model(predictions, probs, train_predictions, train_probs, test_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}

    baseline["recall"] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline["precision"] = precision_score(
        test_labels, [1 for _ in range(len(test_labels))]
    )
    baseline["roc"] = 0.5

    results = {}

    results["recall"] = recall_score(test_labels, predictions)
    results["precision"] = precision_score(test_labels, predictions)
    results["roc"] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results["recall"] = recall_score(train_labels, train_predictions)
    train_results["precision"] = precision_score(train_labels, train_predictions)
    train_results["roc"] = roc_auc_score(train_labels, train_probs)

    for metric in ["recall", "precision", "roc"]:
        print(
            f"{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}"
        )

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, "b", label="baseline")
    plt.plot(model_fpr, model_tpr, "r", label="model")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.show()

def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Oranges
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=20,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("True label", size=18)
    plt.xlabel("Predicted label", size=18)

