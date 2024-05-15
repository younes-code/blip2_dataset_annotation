import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_caption(caption):
    # Lowercase the caption
    caption = caption.lower()

    # Tokenize the caption
    tokens = word_tokenize(caption)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Reconstruct caption
    cleaned_caption = ' '.join(tokens)

    return cleaned_caption

def read_captions(file_path):
    captions_dict = {}
    valid_names = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    pattern = r'^(' + '|'.join(valid_names) + ')'
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
        print ("line_count",line_count)

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                category_name = match.group(1)
            video_name, caption = line.split(':', 1)  # Split at the first colon
            video_name = video_name.strip()
            caption = caption.strip()

            # Preprocess caption
            cleaned_caption = preprocess_caption(caption)

            captions_dict[video_name] = [cleaned_caption]

    return captions_dict

def generate_embeddings(captions_dict, bert_model, tokenizer, output_file):
    embeddings_dict = {}
    total_videos = len(captions_dict)
    print(f"Total videos to process: {total_videos}")
    processed_videos = 0
    
    for video_name, caption in tqdm(captions_dict.items(), desc="Generating embeddings", unit="videos"):
        print(f"Processing video: {video_name}")
        # Tokenize the caption
        inputs = tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            # Get embeddings for the caption
            model_output = bert_model(**inputs)
            caption_embeddings = mean_pooling(model_output, attention_mask)
            
        embeddings_dict[video_name] = caption_embeddings.numpy()
        processed_videos += 1
        print(f"Processed videos: {processed_videos}/{total_videos}")

    # Save embeddings to file
    np.savez(output_file, **embeddings_dict)
    print(f"Embeddings saved to {output_file}")

    # Print the keys of the saved embeddings
    print("Keys in the saved embeddings:")
    with np.load(output_file) as saved_embeddings:
        print(saved_embeddings.keys())


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # Get the embeddings of all tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # Expand attention mask
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)  # Sum the embeddings
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # Sum of attention mask
    return sum_embeddings / sum_mask

def load_pretrained_bert_model():
    print("Starting the download")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Download done")
    return model, tokenizer

if __name__ == '__main__':
    file_path = "concatenated_captions copy.txt"
    output_file = "captions_embeddings.npz"
    
    captions_dict = read_captions(file_path)
    bert_model, tokenizer = load_pretrained_bert_model()
    generate_embeddings(captions_dict, bert_model, tokenizer, output_file)
