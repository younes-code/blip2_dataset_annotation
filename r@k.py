import re

def map_class_name(class_name):
    # Define special cases mapping
    special_cases = {
        'RoadAccidents': 'Road Accidents',
        'Normal_Videos_': 'Normal Videos',
        # Add more special cases as needed
    }

    # Check if the class_name is in special cases
    if class_name in special_cases:
        return special_cases[class_name]
    else:
        return class_name

def calculate_r_at_k(predictions, ground_truth, k=5):
    top_k_predictions = predictions[:k]
    return int(ground_truth in [class_name for class_name, _ in top_k_predictions])

def process_file(file_path, k=5):
    total_r_at_k = 0
    num_lines = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Split the line into class name and predictions
        parts = line.split(':')
        match_object = re.match(r'[^0-9]+', parts[0])

        if match_object:
            raw_class_name = match_object.group(0).strip()
            class_name = map_class_name(raw_class_name)

            # Extract predictions and ground truth from the line
            _, predictions_text = parts[0], parts[1]
            predictions = eval(predictions_text)
            ground_truth = class_name

            # Calculate R@K
            r_at_k = calculate_r_at_k(predictions, ground_truth, k)
            print(f"R@{k} for {class_name}: {r_at_k}")

            total_r_at_k += r_at_k
            num_lines += 1

    mean_r_at_k = total_r_at_k / num_lines if num_lines > 0 else 0
    print(f"Mean R@{k} for all predictions: {mean_r_at_k}")

if __name__ == "__main__":
    file_path = "similarities_by_frame.txt"
    process_file(file_path, k=10)
