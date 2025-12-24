"""
concatenate_captions.py

Script for concatenating captions that belongs to te same video source.

"""


import re
from collections import defaultdict

def read_file(file_path):
    """Read the content of the text file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def extract_source_and_caption(line):
    """
    Extract source and caption from a line.
    Handles formats like 'RoadAccidents055_x264/1 00:00:000.jpg ## caption </s>'.
    """
    # Split line into frame info and caption
    parts = line.split(' ## ')
    if len(parts) != 2:
        return None, None
    
    frame_info, caption = parts
    # Extract source using regex to match format like 'source/frame 00:00:000.jpg'
    match = re.match(r'(.+?)/\d+ \d+:\d+:\d+\.jpg', frame_info)
    if not match:
        return None, None
    
    source = match.group(1).strip()  # e.g., RoadAccidents055_x264
    # Clean caption: remove </s>, <s>, and unwanted characters
    caption = re.sub(r'</?s>', '', caption).strip()
    caption = re.sub(r'[<>,/]', '', caption).strip()
    
    return source, caption

def concatenate_captions(lines):
    """
    Concatenate all captions for each video source into a single string.
    """
    source_captions = defaultdict(list)
    
    for line in lines:
        source, caption = extract_source_and_caption(line)
        if source and caption:  # Only add valid source-caption pairs
            source_captions[source].append(caption)
    
    # Concatenate all captions for each source into one string
    concatenated_captions = {
        source: ' '.join(captions) for source, captions in source_captions.items()
    }
    
    return concatenated_captions

def main(file_path, output_path):
    """Main function to read, concatenate, and write captions."""
    lines = read_file(file_path)
    result = concatenate_captions(lines)
    
    with open(output_path, 'w') as output_file:
        for source, caption in result.items():
            output_file.write(f"{source}: {caption}\n")

if __name__ == "__main__":
    file_path = 'resampled_ucf_captions/captions_interval_5s.txt'
    output_path = 'resampled_ucf_captions/captions_interval_5s_concatenated.txt'
    main(file_path, output_path)