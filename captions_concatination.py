"""
concatenate_captions.py

Script for concatenating captions that belongs to te same video source.

"""

from collections import defaultdict

def get_unique_sources(lines):
    """Get the set of unique source names."""
    unique_sources = set()

    for line in lines:
        source, _ = extract_source_and_caption(line)
        if source is not None:
            unique_sources.add(source)

    return unique_sources

def read_file(file_path):
    """Read the content of the text file"""
    with open(file_path,'r') as file :
        lines=file.readlines()
    return lines

def extract_source_and_caption(line):
      """Extract source and caption from a line."""
      source_split=line.split(' ') #/
      caption_split=line.split('##')

      source,caption=source_split[0].strip(),caption_split[1].strip()
      return source,caption

def concatenate_captions(lines):
      source_captions = defaultdict(list)

      """Concatenate captions for each video."""
      for line in lines:
          source,caption=extract_source_and_caption(line)

          if source is not None:
            source_captions[source].append(caption)

      concatenated_captions = {source: ' ## '.join(captions) for source, captions in source_captions.items()}
      return concatenated_captions
                

def main(file_path,output_path):
    lines = read_file(file_path)
    result = concatenate_captions(lines)
    
    with open (output_path,'w') as output_file:
      for source, captions in result.items():
            output_file.write(f"{source}: {captions}\n")

        
if __name__ == "__main__":
    file_path = 'UCFCrime_Train.txt'
    output_path= 'concatenated_UCFCrime_Train.txt'
    main(file_path,output_path)
