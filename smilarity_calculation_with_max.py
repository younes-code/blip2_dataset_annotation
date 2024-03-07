"""
caption_similarity.py

Script for calculating cosine similarity between video captions and calsses using a pre-trained BERT model.
It takes in consediration each frame from the video and then picks the MAX

"""
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

def read_captions(file_path):
    """
    Reads captions from a file and processes them.

    Args:
        file_path (str): The path to the file containing captions.

    Returns:
        dict: A dictionary mapping video names to a list of processed captions.
    """
    captions_dict = defaultdict(list)
    stop_words = set(stopwords.words('english'))

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        captions = line.split('##')
        name_parts = captions[0].strip().split('_')

        # Handling the exception for video names like 'Normal_Videos'
        if 'Normal' in name_parts:
            video_name = '_'.join(name_parts[:2])
        else:
            video_name = name_parts[0]

        caption_text = captions[1].strip()

        # Removing stop words and converting to lowercase
        cleaned_caption = ' '.join([word.lower().strip() for word in caption_text.split() if word.lower().strip() not in stop_words])

        # Print statements to understand the issue
        # print(f"Original Video Name: {captions[0].strip()}")
        # print(f"Extracted Video Name: {video_name}")
        # print(f"Cleaned Caption: {cleaned_caption}")
        # print()

        captions_dict[video_name].append(cleaned_caption)

    return captions_dict


def load_pretrained_bert_model():
    """
    Loads a pre-trained BERT model and tokenizer.

    Returns:
        tuple: A tuple containing the loaded BERT model and tokenizer.
    """
    print("Starting the download")

    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Download done")
    return model, tokenizer

import torch.nn.functional as F

def calculate_similarity(captions, target_words, bert_model, tokenizer):
    """
    Calculates cosine similarity between captions and classes.

    Args:
        captions (dict): A dictionary mapping video names to processed captions.
        target_words (list): A list of classes for similarity calculation.
        bert_model (AutoModel): The pre-trained BERT model.
        tokenizer (AutoTokenizer): The BERT tokenizer.

    Returns:
        dict: A dictionary mapping video names to a list of maximum similarities.
    """
    similarities = {}
    for video_name, captions_list in captions.items():
        # Dictionary to store the maximum similarity for each target word for the current video
        max_similarity_dict = {target_word: -1 for target_word in target_words}

        for caption in tqdm(captions_list):
            # Tokenize the caption and classes
            caption_tokens = tokenizer(caption, return_tensors="pt", truncation=True, padding=True)['input_ids']
            target_word_tokens = [tokenizer(word, return_tensors="pt", truncation=True, padding=True)['input_ids'] for word in target_words]

            # Ensure the shape of caption_tokens is (batch_size, seq_length)
            caption_tokens = caption_tokens.view(1, -1) if len(caption_tokens.shape) == 1 else caption_tokens

            # Get embeddings for the caption
            with torch.no_grad():
                caption_outputs = bert_model(input_ids=caption_tokens)
                caption_embeddings = caption_outputs.last_hidden_state[:, 0, :]

            # Get embeddings for the target words
            target_word_embeddings = []
            for word in target_word_tokens:
                # Ensure the shape of target_word_tokens is (batch_size, seq_length)
                word = word.view(1, -1) if len(word.shape) == 1 else word
                with torch.no_grad():
                    target_outputs = bert_model(input_ids=word)
                    target_word_embeddings.append(target_outputs.last_hidden_state[:, 0, :])

            # Calculate cosine similarity between the caption and each target word
            similarities_list = [(target_words[i], cosine_similarity(caption_embeddings, target_word_embedding)[0][0])
                                for i, target_word_embedding in enumerate(target_word_embeddings)]

            # Apply softmax to the similarities
            softmax_values = F.softmax(torch.tensor([similarity for _, similarity in similarities_list]), dim=0)
            softmax_similarities = [(target_word, float(softmax_values[i])) for i, (target_word, _) in enumerate(similarities_list)]

            # Print softmax similarities for each caption before aggregation
            print(f"Video: {video_name}, Caption: {caption}")
            print("Softmax Similarities:")
            for target_word, softmax_similarity in softmax_similarities:
                print(f"{target_word}: {softmax_similarity}")

            # Update the maximum similarity for each target word
            for target_word, softmax_similarity in softmax_similarities:
                max_similarity_dict[target_word] = max(max_similarity_dict[target_word], softmax_similarity)

        # Append the final result for the current video to the similarities dictionary
        similarities[video_name] = list(max_similarity_dict.items())

    return similarities





def main(file_path, target_words, output_file):
    """
    Main function for calculating caption similarities and saving results.

    Args:
        file_path (str): The path to the file containing captions.
        target_words (list): A list of target words for similarity calculation.
        output_file (str): The path to the output file for saving results.

    Returns:
        None
    """
    captions = read_captions(file_path)
    # Load pre-trained BERT model
    bert_model, tokenizer = load_pretrained_bert_model()
    similarities = calculate_similarity(captions, target_words, bert_model, tokenizer)

    with open(output_file, 'w') as out_file:
        for video_name, similarity_list in similarities.items():
            result_str = f"{video_name}: {sorted(similarity_list, key=lambda x: x[1], reverse=True)}"
            print(result_str)
            out_file.write(result_str + '\n')

if __name__ == '__main__':
    file_path = "captions.txt"
    output_file = "similarities_with_max_softmax.txt"
    classes = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal Videos", "Road Accidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    main(file_path, classes, output_file)

