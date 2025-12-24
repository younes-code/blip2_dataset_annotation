# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import re
# from nltk.corpus import stopwords

# def read_captions(file_path):
#     captions_dict = {}
#     valid_names = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
#     pattern = r'^(' + '|'.join(valid_names) + r')([^/]*)'

#     with open(file_path, 'r') as file:
#         for line in file:
#             match = re.match(pattern, line)
#             if match:
#                 category_name = match.group(1)
#             video_name, caption = line.split(':', 1)  # Split at the first colon
#             video_name = video_name.strip()
#             caption = caption.strip()

#             # Removing stop words and converting to lowercase
#             stop_words = set(stopwords.words('english'))
#             cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])

#             captions_dict[video_name] = [cleaned_caption]

#     return captions_dict

# def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings, target_words):
#     """
#     Calculates similarity between pre-computed embeddings and target words.

#     Args:
#         embeddings_dict (dict): A dictionary mapping video names to pre-computed embeddings.
#         target_word_embeddings (np.array): An array containing embeddings for target words.
#         target_words (list): A list of target words for similarity calculation.

#     Returns:
#         dict: A dictionary mapping video names to a list of similarities.
#     """
#     similarities = {}
#     for video_name, embeddings in embeddings_dict.items():
#         for caption_embeddings in embeddings:
#             # Ensure caption_embeddings is a 2D array
#             if len(caption_embeddings.shape) == 1:
#                 caption_embeddings = np.expand_dims(caption_embeddings, axis=0)

#             similarities_list = []
#             for i in range(len(target_words)):
#                 # Ensure target_word_embeddings[i] is a 2D array
#                 target_word_embedding_i = np.atleast_2d(target_word_embeddings[i])
                
#                 # Calculate cosine similarity between the caption and the target word
#                 similarity = cosine_similarity(caption_embeddings, target_word_embedding_i)
                
#                 # Append the similarity to the list
#                 similarities_list.append((target_words[i], similarity[0][0]))

#             # Sort the similarities in descending order
#             sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)

#             if video_name not in similarities:
#                 similarities[video_name] = []

#             similarities[video_name].append({
#                 'similarities': sorted_similarities
#             })

#     return similarities


# def get_target_word_embeddings(target_words, model):
#     """
#     Obtain embeddings for target words using SentenceTransformer model.

#     Args:
#         target_words (list): A list of target words.
#         model (SentenceTransformer): The SentenceTransformer model.

#     Returns:
#         np.array: An array containing embeddings for target words.
#     """
#     target_word_embeddings = []
#     for word in target_words:
#         # Encode the target word
#         word_embedding = model.encode([word])[0]
#         # print(word,word_embedding.shape)
#         target_word_embeddings.append(word_embedding)
#         # print(word,target_word_embeddings)
        
#     return target_word_embeddings

# def main_calculate_similarity(file_path, embeddings_file, target_words, output_file):
#     """
#     Main function for calculating caption similarities using pre-computed embeddings.

#     Args:
#         file_path (str): The path to the file containing captions.
#         embeddings_file (str): The path to the file containing pre-computed embeddings.
#         target_words (list): A list of target words for similarity calculation.
#         output_file (str): The path to the output file for saving results.

#     Returns:
#         None
#     """
#     # Load embeddings from the file
#     embeddings_dict = np.load(embeddings_file)
#     print("Embeddings file contents:")
#     print(len(embeddings_dict.keys()))
#     print(embeddings_dict["RoadAccidents055_x264"].shape)
#     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#     # Get target word embeddings
#     target_word_embeddings = get_target_word_embeddings(target_words, model)

#     similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings, target_words)

#     with open(output_file, 'w') as out_file:
#         for video_name, similarity_list in similarities.items():
#             result_str = f"{video_name}: {sorted(similarity_list[0]['similarities'], key=lambda x: x[1], reverse=True)}"
#             # print(result_str)
#             out_file.write(result_str + '\n')
# if __name__ == '__main__':
#     file_path = "concatenated_captions.txt"
#     embeddings_file = "captions_embeddings.npz"
#     output_file = "similarities.txt"
#     # target_words=['Abuse','RoadAccidents','Normal_Videos_','Burglary','Explosion','Shoplifting','Robbery']
#     target_words = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting","Accident", "Normal Videos", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
#     main_calculate_similarity(file_path, embeddings_file, target_words, output_file)

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
from nltk.corpus import stopwords

def read_captions(file_path):
    captions_dict = {}
    valid_names = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    pattern = r'^(' + '|'.join(valid_names) + r')([^/]*)'

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                category_name = match.group(1)
            video_name, caption = line.split(':', 1)  # Split at the first colon
            video_name = video_name.strip()
            caption = caption.strip()

            # Removing stop words and converting to lowercase
            stop_words = set(stopwords.words('english'))
            cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])

            captions_dict[video_name] = [cleaned_caption]

    return captions_dict

def get_target_word_embeddings(target_word_lists, model):
    """
    Obtain embeddings for target words using SentenceTransformer model.

    Args:
        target_word_lists (list): A list of strings, each containing a target word and related words.
        model (SentenceTransformer): The SentenceTransformer model.

    Returns:
        list: A list of tuples containing target words and their embeddings.
    """
    target_word_embeddings = []
    for words in target_word_lists:
        word_list = words.split(',')
        target_word = word_list[0].strip()  # Ensure the target word is stripped of any leading/trailing spaces
        sentence = ' '.join(word_list)
        # Encode the sentence
        sentence_embedding = model.encode([sentence])[0]
        target_word_embeddings.append((target_word, sentence_embedding))
        
    return target_word_embeddings


def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings):
    """
    Calculates similarity between pre-computed embeddings and target word sentences.

    Args:
        embeddings_dict (dict): A dictionary mapping video names to pre-computed embeddings.
        target_word_embeddings (list): A list of tuples containing target words and their embeddings.

    Returns:
        dict: A dictionary mapping video names to a list of similarities.
    """
    similarities = {}
    for video_name, embeddings in embeddings_dict.items():
        for caption_embeddings in embeddings:
            if len(caption_embeddings.shape) == 1:
                caption_embeddings = np.expand_dims(caption_embeddings, axis=0)

            similarities_list = []
            for target_word, target_word_embedding in target_word_embeddings:
                target_word_embedding = np.atleast_2d(target_word_embedding)
                similarity = cosine_similarity(caption_embeddings, target_word_embedding)
                similarities_list.append((target_word, similarity[0][0]))

            sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)
            if video_name not in similarities:
                similarities[video_name] = []

            similarities[video_name].append({
                'similarities': sorted_similarities
            })

    return similarities

def main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file):
    """
    Main function for calculating caption similarities using pre-computed embeddings.

    Args:
        file_path (str): The path to the file containing captions.
        embeddings_file (str): The path to the file containing pre-computed embeddings.
        target_word_lists (list): A list of strings, each containing a target word and related words.
        output_file (str): The path to the output file for saving results.

    Returns:
        None
    """
    # Load embeddings from the file
    embeddings_dict = np.load(embeddings_file)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Get target word embeddings
    target_word_embeddings = get_target_word_embeddings(target_word_lists, model)
    similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings)

    with open(output_file, 'w') as out_file:
        for video_name, similarity_list in similarities.items():
            result_str = f"{video_name}: {sorted(similarity_list[0]['similarities'], key=lambda x: x[1], reverse=True)}"
            out_file.write(result_str + '\n')


if __name__ == '__main__':
    file_path = "captions_interval_7s_concatenated.txt"
    embeddings_file = "embeddings/ucf_7s_concatenated_captions_embeddings.npz"
    output_file = "similarities/interval_7s_similarities.txt"
    target_word_lists = ["Abuse,Abuse,maltreatment, mistreatment, cruelty, harm, exploitation, violence, harassment, injury, oppression, torment", 
                         "Arrest,detain, apprehend, capture, take into custody, seize,law enforcement, police, handcuffs, booking, incarceration, detention", 
                         "Arson,fire-raising, incendiarism, pyromania, torching,fire, blaze, ignition, burning, conflagration, fire-setting", 
                         "Assault,attack, physical attack, aggression,violence, strike", 
                         "Burglary,break-in, robbery, housebreaking, theft,intrusion, trespassing, stealing, larceny, heist, pilfering", 
                         "Explosion,blast, detonation, eruption, blowup,bomb, burst, combustion, detonate, rupture, outburst", 
                         "Fighting,brawling, combat, conflict, struggle, tussle,altercation, skirmish, clash, scuffle, fray, melee",
                         "Accident, accident ", 
                         "Normal Videos,routine, ordinary, standard, typical, regular,everyday, usual, common, mundane, conventional, normalcy", 
                         "Robbery,theft, heist, larceny, hold-up, mugging,burglary, stealing, pilfering, banditry, thievery, looting", 
                         "Shooting,gunfire, gunfight, gunplay, firing, discharge,firearm, bullet, shootout, sniper, marksman, ballistics", 
                         "Shoplifting,theft, pilfering, larceny, stealing,shop theft, retail theft, snatching, shop burglarizing, petty theft", 
                         "Stealing,theft, larceny, robbery, pilfering, thievery,embezzlement, burglary, shoplifting, swiping, looting, filching", 
                         "Vandalism,Vandalism,destruction, defacement, sabotage,graffiti, wrecking, ruin"]
    main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file)
