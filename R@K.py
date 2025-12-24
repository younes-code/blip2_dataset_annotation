import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    unique_labels_list = unique_labels(y_true, y_pred)
    classes_dict = {label: idx for idx, label in enumerate(unique_labels_list)}
    classes = [classes_dict[label] for label in classes]
    class_names = [label for label in unique_labels_list]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print("Classes:", class_names)  # Debugging statement

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=[f"{class_names[i]} ({i})" for i in classes],  # Include class names and indices on x-axis
           yticklabels=[f"{class_names[i]} ({i})" for i in classes],  # Include class names and indices on y-axis
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def map_class_name(class_name):
    if class_name.startswith("RoadAccidents"):
        # Replace "RoadAccidents" with "Accident"
        return class_name.replace("RoadAccidents", "Accident")
    elif class_name.startswith("Normal_Videos_"):
        # Replace "Normal_Videos_" with "Normal Videos"
        return class_name.replace("Normal_Videos_", "Normal Videos")
    else:
        return class_name
    # return class_name
    
def calculate_r_at_k(predictions, ground_truth, k=5):
    top_k_predictions = predictions[:k]
    return int(ground_truth in [class_name for class_name, _ in top_k_predictions])

def process_file(file_path, k=10):
    total_r_at_k = 0
    num_lines = 0
    y_true = []
    y_pred = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
    print(lines[0])
    for line in lines:
        # Split the line into class name and predictions
        parts = line.split(':')
        match_object = re.match(r'[^0-9]+', parts[0])

        if match_object:
            raw_class_name = match_object.group(0).strip()
            class_name = map_class_name(raw_class_name)

            # Debugging: Print class names after mapping
            # print("Mapped Class Name:", class_name)

            # Extract predictions and ground truth from the line
            _, predictions_text = parts[0], parts[1]
            predictions = eval(predictions_text)
            ground_truth = class_name

            # Debugging: Print predictions and ground truth labels
            # print("Ground Truth:", ground_truth)
            # print("Predictions:", predictions)

            # Calculate R@K
            r_at_k = calculate_r_at_k(predictions, ground_truth, k)
            # print(f"R@{k} for {class_name}: {r_at_k}")

            total_r_at_k += r_at_k
            num_lines += 1

            # Append true and predicted labels
            y_true.append(ground_truth)
            y_pred.append(predictions[0][0])  # Considering only the top-1 prediction

    mean_r_at_k = total_r_at_k / num_lines if num_lines > 0 else 0
    print(f"Mean R@{k} for all predictions: {mean_r_at_k}")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=np.unique(y_true), normalize=True, title='Normalized confusion matrix')
    plt.show()

if __name__ == "__main__":
    file_path = "similarities/interval_5s_similarities.txt"
    process_file(file_path, k=1)
