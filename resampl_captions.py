import re
from collections import defaultdict
import os

def parse_timestamp(timestamp):
    """
    Parse timestamp in format 'MM:SS:MMM' to seconds.
    Example: '00:00:000' -> 0.0 seconds, '00:01:366' -> 1.366 seconds
    """
    match = re.match(r'(\d+):(\d+):(\d+)', timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    milliseconds = int(match.group(3))
    return minutes * 60 + seconds + milliseconds / 1000.0

def group_captions_by_source(file_path):
    """
    Read captions from file and group by video source in chronological order.
    Returns a dictionary mapping source to list of (timestamp, frame_info, caption).
    """
    captions_by_source = defaultdict(list)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Split line into frame info and caption
        parts = line.split(' ## ')
        if len(parts) != 2:
            print(f"Skipping malformed line: {line.strip()}")
            continue
            
        frame_info, caption = parts
        # Extract source and timestamp using flexible regex
        match = re.match(r'(.+?)/\d+ (\d+:\d+:\d+)\.jpg', frame_info)
        if not match:
            print(f"Skipping line with invalid frame info: {frame_info}")
            continue
            
        source = match.group(1)  # e.g., RoadAccidents055_x264 or abnormal_scene_21_scenario_3
        timestamp = match.group(2)  # e.g., 00:00:000 or 00:00:233
        try:
            time_in_seconds = parse_timestamp(timestamp)
            captions_by_source[source].append((time_in_seconds, frame_info, caption.strip()))
        except ValueError as e:
            print(f"Error parsing timestamp in line: {line.strip()}. Error: {e}")
            continue
    
    # Sort captions by timestamp within each source
    for source in captions_by_source:
        captions_by_source[source].sort(key=lambda x: x[0])
    
    return captions_by_source

def select_captions_by_skipping(captions_by_source, input_interval_seconds, target_interval_seconds):
    """
    Select captions by skipping to approximate the target interval.
    input_interval_seconds: Time between consecutive captions in input (e.g., 3.0 for new example).
    target_interval_seconds: Desired time interval (e.g., 3.0, 5.0).
    Returns a list of (frame_info, caption) tuples.
    """
    selected_captions = []
    
    # Calculate number of captions to skip
    skip_count = max(1, round(target_interval_seconds / input_interval_seconds))
    
    for source, captions in captions_by_source.items():
        # Select every skip_count-th caption
        for i in range(0, len(captions), skip_count):
            time, frame_info, caption = captions[i]
            selected_captions.append((frame_info, caption))
    
    # Sort by frame_info to maintain original order across sources
    selected_captions.sort(key=lambda x: x[0])
    return selected_captions

def write_caption_file(captions, output_path):
    """
    Write selected captions to a new file.
    """
    with open(output_path, 'w') as f:
        for frame_info, caption in captions:
            f.write(f"{frame_info} ## {caption}\n")
    print(f"Captions written to {output_path}")

def resample_captions(input_file, output_dir, input_interval_seconds, target_intervals_seconds):
    """
    Main function to resample captions by skipping to achieve target intervals.
    input_interval_seconds: Time between consecutive captions in input (e.g., 3.0 for new example).
    target_intervals_seconds: List of desired intervals (e.g., [3, 5]).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group captions by source
    captions_by_source = group_captions_by_source(input_file)
    
    # Generate new caption files for each target interval
    for target_interval in target_intervals_seconds:
        selected_captions = select_captions_by_skipping(
            captions_by_source, input_interval_seconds, target_interval
        )
        output_file = os.path.join(output_dir, f"captions_interval_{target_interval}s.txt")
        write_caption_file(selected_captions, output_file)

if __name__ == "__main__":
    input_file = "captions.txt"
    output_dir = "resampled_ucf_captions"
    input_interval_seconds = 3.0  # Adjust based on your input file's sampling rate
    target_intervals_seconds = [5, 7]  # Desired sampling intervals
    resample_captions(input_file, output_dir, input_interval_seconds, target_intervals_seconds)